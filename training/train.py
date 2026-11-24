"""Train the SkyEar DS-CNN, quantise it to int8 and export it for the ESP32-S3.

Data layout (one folder per class, any sample rate, mono or stereo WAV):
    data/background/*.wav   wind, distant traffic, birds, room noise
    data/drone/*.wav        quadcopter rotor audio
    data/speech/*.wav       people talking
    data/vehicle/*.wav      cars, motorcycles, trucks

Usage:
    python train.py --data data --out ../node/src/model_data.h

Writes:
    ../node/src/model_data.h    the int8 model as a C array (replaces the placeholder)
    results/metrics.json        dataset counts, accuracy, false-alarm rate, model size
    results/confusion.txt       confusion matrix on the held-out test files
    results/skyear_int8.tflite  the quantised model

Splits are made by FILE, not by window, so overlapping windows from one recording
never land in both train and test.
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

from features import SAMPLE_RATE, WINDOW_SAMPLES, N_FRAMES, N_MELS, logmel

CLASSES = ["background", "drone", "speech", "vehicle"]  # index order baked into the firmware
DRONE = CLASSES.index("drone")
HOP = WINDOW_SAMPLES // 2  # 0.5 s, same as the node
ALARM_P = 0.8


def load_wav(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


def windows(audio):
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, HOP):
        yield audio[start:start + WINDOW_SAMPLES]


def augment(w, rng):
    # Random gain (+-12 dB) covers the level gap between public recordings and the INMP441;
    # light noise and time shift make the model less brittle.
    if rng.random() < 0.5:
        # Simulates playback through a phone/laptop speaker (weak bass, rolled-off top).
        hp = butter(4, rng.uniform(150, 450), "highpass", fs=SAMPLE_RATE, output="sos")
        lp = butter(4, rng.uniform(3500, 7000), "lowpass", fs=SAMPLE_RATE, output="sos")
        w = sosfilt(lp, sosfilt(hp, w)).astype(np.float32)
    w = w * (10 ** (rng.uniform(-12, 12) / 20))
    w = w + rng.normal(0, 10 ** (rng.uniform(-70, -45) / 20), size=w.shape)
    w = np.roll(w, rng.integers(-1600, 1600))
    return np.clip(w, -1, 1).astype(np.float32)


def build_set(files, rng, n_aug):
    X, y = [], []
    for path, label in files:
        for w in windows(load_wav(path)):
            X.append(logmel(w)); y.append(label)
            for _ in range(n_aug):
                X.append(logmel(augment(w, rng))); y.append(label)
    return np.array(X, dtype=np.float32)[..., None], np.array(y, dtype=np.int64)


def split_files(data_dir, seed):
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    counts = {}
    for label, name in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(data_dir, name, "**", "*.wav"), recursive=True))
        if not files:
            raise SystemExit(f"no WAV files in {os.path.join(data_dir, name)}")
        rng.shuffle(files)
        n_val = max(1, round(len(files) * 0.15))
        n_test = max(1, round(len(files) * 0.15))
        parts = {"test": files[:n_test], "val": files[n_test:n_test + n_val], "train": files[n_test + n_val:]}
        for k, v in parts.items():
            splits[k] += [(f, label) for f in v]
        counts[name] = {k: len(v) for k, v in parts.items()}
    return splits, counts


def build_model(tf, batch_size=None):
    L = tf.keras.layers
    # momentum 0.9 (default 0.99) so BatchNorm running stats converge even on small
    # datasets; with 0.99 a few hundred steps leave eval-mode stats near their init.
    BN = lambda: L.BatchNormalization(momentum=0.9)
    inp = L.Input(shape=(N_FRAMES, N_MELS, 1), batch_size=batch_size)
    x = L.Conv2D(32, (10, 4), strides=(2, 2), padding="same", use_bias=False)(inp)
    x = BN()(x)
    x = L.ReLU()(x)
    for _ in range(4):  # depthwise-separable blocks
        x = L.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = BN()(x)
        x = L.ReLU()(x)
        x = L.Conv2D(32, (1, 1), use_bias=False)(x)
        x = BN()(x)
        x = L.ReLU()(x)
    x = L.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
    x = L.Flatten()(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inp, out, name="skyear_ds_cnn")


def to_int8(tf, model, X_rep):
    # Re-build with a fixed batch of 1 so Flatten exports as one static RESHAPE instead of
    # SHAPE / STRIDED_SLICE / PACK ops that only waste time on the microcontroller.
    export = build_model(tf, batch_size=1)
    export.set_weights(model.get_weights())
    conv = tf.lite.TFLiteConverter.from_keras_model(export)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep():
        for i in np.random.default_rng(0).choice(len(X_rep), size=min(400, len(X_rep)), replace=False):
            yield [X_rep[i:i + 1]]

    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(tf, tflite_bytes, X):
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = i["quantization"]
    s_out, z_out = o["quantization"]
    probs = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    for n in range(len(X)):
        q = np.clip(np.round(X[n:n + 1] / s_in) + z_in, -128, 127).astype(np.int8)
        it.set_tensor(i["index"], q)
        it.invoke()
        probs[n] = (it.get_tensor(o["index"])[0].astype(np.float32) - z_out) * s_out
    return probs


def report(probs, y):
    pred = probs.argmax(1)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    is_drone = y == DRONE
    fires = probs[:, DRONE] >= ALARM_P
    return {
        "accuracy": float((pred == y).mean()),
        "drone_recall_at_0.8": float(fires[is_drone].mean()) if is_drone.any() else None,
        "false_alarm_rate_at_0.8": float(fires[~is_drone].mean()) if (~is_drone).any() else None,
        "test_windows": int(len(y)),
    }, cm


def write_header(path, tflite_bytes):
    lines = [
        "// GENERATED by training/train.py. Do not edit by hand.",
        "#pragma once",
        "#include <stdint.h>",
        "#define SKYEAR_HAS_MODEL 1",
        f"#define SKYEAR_NUM_CLASSES {len(CLASSES)}",
        f"#define SKYEAR_DRONE_INDEX {DRONE}",
        "static const char* const kSkyEarClassNames[SKYEAR_NUM_CLASSES] = {"
        + ", ".join(f'"{c}"' for c in CLASSES) + "};",
        f"static const unsigned int g_skyear_model_len = {len(tflite_bytes)};",
        "alignas(16) static const unsigned char g_skyear_model[] = {",
    ]
    for i in range(0, len(tflite_bytes), 16):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i + 16]) + ",")
    lines.append("};")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="../node/src/model_data.h")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--aug", type=int, default=2, help="augmented copies per training window")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs("results", exist_ok=True)

    splits, file_counts = split_files(args.data, args.seed)
    print("files per class:", json.dumps(file_counts))
    Xtr, ytr = build_set(splits["train"], rng, args.aug)
    Xva, yva = build_set(splits["val"], rng, 0)
    Xte, yte = build_set(splits["test"], rng, 0)
    print(f"windows: train {len(ytr)}  val {len(yva)}  test {len(yte)}")

    counts = np.bincount(ytr, minlength=len(CLASSES))
    class_weight = {i: float(len(ytr) / (len(CLASSES) * max(c, 1))) for i, c in enumerate(counts)}

    model = build_model(tf)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=64,
              class_weight=class_weight, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)])

    float_metrics, _ = report(model.predict(Xte, verbose=0), yte)
    tflite_bytes = to_int8(tf, model, Xtr)
    int8_metrics, cm = report(run_tflite(tf, tflite_bytes, Xte), yte)

    with open("results/skyear_int8.tflite", "wb") as f:
        f.write(tflite_bytes)
    write_header(args.out, tflite_bytes)

    metrics = {
        "classes": CLASSES,
        "files_per_class": file_counts,
        "windows": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))},
        "params": int(model.count_params()),
        "int8_model_bytes": len(tflite_bytes),
        "float_test": float_metrics,
        "int8_test": int8_metrics,
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("results/confusion.txt", "w") as f:
        f.write("rows = true, cols = predicted\n" + " " * 12 + "".join(f"{c[:10]:>12}" for c in CLASSES) + "\n")
        for name, row in zip(CLASSES, cm):
            f.write(f"{name[:10]:>12}" + "".join(f"{v:>12}" for v in row) + "\n")
    print(json.dumps(metrics, indent=2))
    print(open("results/confusion.txt").read())
    print(f"wrote {args.out} ({len(tflite_bytes)} bytes)")


if __name__ == "__main__":
    main()
