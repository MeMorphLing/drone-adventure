"""Builds training/data/<class>/ from the public datasets.

    python prep_data.py --drone <DroneAudioDataset folder> --esc50 <ESC-50 folder> [--speech <your recordings>]

drone      : DroneAudioDataset  Binary_Drone_Audio/yes_drone
background : DroneAudioDataset  Binary_Drone_Audio/unknown (sampled) + ESC-50 nature/indoor sounds
speech     : ESC-50 human sounds + any WAVs in --speech (record yourself talking)
vehicle    : ESC-50 engines, cars, trains, aircraft, helicopter (hard negatives for drones)
"""
import argparse, csv, glob, os, random, shutil

ESC_MAP = {
    "background": ["wind", "rain", "crickets", "chirping_birds", "insects", "sea_waves", "water_drops",
                   "clock_tick", "keyboard_typing", "footsteps", "thunderstorm", "vacuum_cleaner",
                   "washing_machine", "mouse_click", "pouring_water", "toilet_flush", "crackling_fire"],
    "speech": ["laughing", "coughing", "sneezing", "crying_baby", "breathing", "snoring", "clapping"],
    "vehicle": ["engine", "car_horn", "train", "airplane", "helicopter", "siren", "chainsaw", "hand_saw"],
}
CAP = {"drone": 1500, "background": 1500, "speech": 1500, "vehicle": 1500}


def find_dir(root, name):
    for d, subdirs, _ in os.walk(root):
        if os.path.basename(d) == name:
            return d
    return None


def copy(files, dst, prefix):
    os.makedirs(dst, exist_ok=True)
    for i, f in enumerate(files):
        shutil.copy(f, os.path.join(dst, f"{prefix}_{i:05d}.wav"))
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone", required=True, help="extracted DroneAudioDataset folder")
    ap.add_argument("--esc50", required=True, help="extracted ESC-50 folder")
    ap.add_argument("--speech", help="folder of your own speech WAVs (optional)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    rng = random.Random(0)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)

    yes = find_dir(a.drone, "yes_drone")
    unk = find_dir(a.drone, "unknown")
    if not yes:
        raise SystemExit("yes_drone folder not found under " + a.drone)
    drone = sorted(glob.glob(os.path.join(yes, "*.wav")))
    rng.shuffle(drone)
    n = {"drone": copy(drone[:CAP["drone"]], os.path.join(a.out, "drone"), "drone")}

    meta = glob.glob(os.path.join(a.esc50, "**", "esc50.csv"), recursive=True)
    if not meta:
        raise SystemExit("meta/esc50.csv not found under " + a.esc50)
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(meta[0])), "audio")
    esc = {k: [] for k in ESC_MAP}
    with open(meta[0]) as fh:
        for row in csv.DictReader(fh):
            for cls, cats in ESC_MAP.items():
                if row["category"] in cats:
                    esc[cls].append(os.path.join(audio_dir, row["filename"]))

    bg = esc["background"]
    if unk:
        u = sorted(glob.glob(os.path.join(unk, "*.wav")))
        rng.shuffle(u)
        bg = bg + u[:max(0, CAP["background"] - len(bg))]
    n["background"] = copy(bg, os.path.join(a.out, "background"), "bg")

    sp = esc["speech"]
    if a.speech:
        sp = sp + sorted(glob.glob(os.path.join(a.speech, "**", "*.wav"), recursive=True))
    n["speech"] = copy(sp, os.path.join(a.out, "speech"), "speech")
    n["vehicle"] = copy(esc["vehicle"], os.path.join(a.out, "vehicle"), "veh")

    for k, v in n.items():
        print(f"{k:11s} {v} files")


if __name__ == "__main__":
    main()
