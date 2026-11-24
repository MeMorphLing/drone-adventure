"""Log-mel features, bit-for-bit the same recipe as node/src/logmel.cpp.

16 kHz mono, 640-sample Hann frames, 320 hop, 1024-point FFT,
40 HTK mel bands 20 Hz - 8 kHz, log(mel + 1e-6). 1 s of audio -> (49, 40).
"""
import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
FRAME_LEN = 640
HOP = 320
FFT_SIZE = 1024
N_MELS = 40
F_MIN, F_MAX = 20.0, 8000.0
LOG_EPS = 1e-6
N_FRAMES = 1 + (WINDOW_SAMPLES - FRAME_LEN) // HOP  # 49


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _filterbank():
    edges = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))
    freqs = np.arange(FFT_SIZE // 2 + 1) * SAMPLE_RATE / FFT_SIZE
    fb = np.zeros((N_MELS, freqs.size))
    for m in range(N_MELS):
        fl, fc, fr = edges[m], edges[m + 1], edges[m + 2]
        rise = (freqs > fl) & (freqs < fc)
        fall = (freqs >= fc) & (freqs < fr)
        fb[m, rise] = (freqs[rise] - fl) / (fc - fl)
        fb[m, fall] = (fr - freqs[fall]) / (fr - fc)
    return fb.astype(np.float32)


_WINDOW = np.hanning(FRAME_LEN).astype(np.float32)
_FB = _filterbank()


def logmel(audio):
    """audio: float array of WINDOW_SAMPLES samples in [-1, 1] -> (49, 40) float32."""
    audio = np.asarray(audio, dtype=np.float32)
    assert audio.shape == (WINDOW_SAMPLES,), audio.shape
    idx = np.arange(FRAME_LEN)[None, :] + HOP * np.arange(N_FRAMES)[:, None]
    frames = audio[idx] * _WINDOW
    spec = np.fft.rfft(frames, n=FFT_SIZE, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(power @ _FB.T + LOG_EPS).astype(np.float32)
