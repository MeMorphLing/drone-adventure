"""Compares the C++ front end against training/features.py on test signals.

Run from the repo root:
  g++ -O2 -std=c++17 -Inode/src tools/test_logmel.cpp node/src/logmel.cpp -o tools/test_logmel
  python tools/check_logmel.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from features import logmel, SAMPLE_RATE, WINDOW_SAMPLES  # noqa: E402

BIN = os.path.join(os.path.dirname(__file__), "test_logmel")
rng = np.random.default_rng(0)
t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
signals = {
    "rotor-like harmonics": sum(0.2 / k * np.sin(2 * np.pi * 180 * k * t) for k in range(1, 8)),
    "white noise": 0.1 * rng.standard_normal(WINDOW_SAMPLES),
    "quiet noise": 1e-4 * rng.standard_normal(WINDOW_SAMPLES),
    "silence": np.zeros(WINDOW_SAMPLES),
}
worst = 0.0
with tempfile.TemporaryDirectory() as d:
    for name, sig in signals.items():
        sig = sig.astype(np.float32)
        src, dst = os.path.join(d, "in.f32"), os.path.join(d, "out.f32")
        sig.tofile(src)
        subprocess.run([BIN, src, dst], check=True)
        c = np.fromfile(dst, dtype=np.float32).reshape(49, 40)
        p = logmel(sig)
        err = np.abs(c - p).max()
        worst = max(worst, err)
        print(f"{name:22s} max abs diff {err:.2e}   (python range {p.min():.2f} .. {p.max():.2f})")
print("PASS" if worst < 1e-2 else "FAIL", f"worst diff {worst:.2e}")
sys.exit(0 if worst < 1e-2 else 1)
