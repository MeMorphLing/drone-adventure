// Log-mel front end. Must match training/features.py exactly:
// 16 kHz, 640-sample (40 ms) Hann frames, 320-sample (20 ms) hop, 1024-point FFT,
// 40 HTK mel bands from 20 Hz to 8 kHz, log(mel + 1e-6). 1 s of audio -> 49 x 40.
// Plain C++ with no Arduino dependency, so tools/test_logmel.cpp can check it on a PC.
#pragma once
#include <stdint.h>

namespace logmel {

constexpr int kSampleRate = 16000;
constexpr int kWindowSamples = 16000;  // 1 s analysis window
constexpr int kFrameLen = 640;
constexpr int kHop = 320;
constexpr int kFftSize = 1024;
constexpr int kNumBins = kFftSize / 2 + 1;  // 513
constexpr int kNumMel = 40;
constexpr int kNumFrames = 1 + (kWindowSamples - kFrameLen) / kHop;  // 49
constexpr float kFMin = 20.0f;
constexpr float kFMax = 8000.0f;
constexpr float kLogEps = 1e-6f;

// Builds the window, twiddles and mel filterbank. Call once.
void init();

// audio: kWindowSamples samples scaled to [-1, 1].
// out:   kNumFrames * kNumMel floats, frame-major (out[f * kNumMel + m]).
void compute(const float* audio, float* out);

}  // namespace logmel
