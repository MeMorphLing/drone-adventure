#include "logmel.h"
#include <math.h>
#include <string.h>

namespace logmel {
namespace {

float g_window[kFrameLen];
float g_cos[kFftSize / 2];
float g_sin[kFftSize / 2];
uint16_t g_bitrev[kFftSize];

// Sparse triangular filters: each band covers bins [start, start + count).
uint16_t g_mel_start[kNumMel];
uint16_t g_mel_count[kNumMel];
float g_mel_weights[kNumMel * 64];  // generous upper bound on bins per band
uint16_t g_mel_offset[kNumMel];

float g_re[kFftSize];
float g_im[kFftSize];
float g_power[kNumBins];

inline double hz_to_mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
inline double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

void fft() {
  for (int i = 0; i < kFftSize; i++) {
    int j = g_bitrev[i];
    if (j > i) {
      float t = g_re[i]; g_re[i] = g_re[j]; g_re[j] = t;
      t = g_im[i]; g_im[i] = g_im[j]; g_im[j] = t;
    }
  }
  for (int len = 2; len <= kFftSize; len <<= 1) {
    int half = len >> 1;
    int step = kFftSize / len;
    for (int i = 0; i < kFftSize; i += len) {
      for (int k = 0; k < half; k++) {
        float wr = g_cos[k * step], wi = -g_sin[k * step];
        int a = i + k, b = a + half;
        float xr = g_re[b] * wr - g_im[b] * wi;
        float xi = g_re[b] * wi + g_im[b] * wr;
        g_re[b] = g_re[a] - xr; g_im[b] = g_im[a] - xi;
        g_re[a] += xr;          g_im[a] += xi;
      }
    }
  }
}

}  // namespace

void init() {
  // Symmetric Hann, same as numpy.hanning(640).
  for (int n = 0; n < kFrameLen; n++)
    g_window[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / (kFrameLen - 1)));

  for (int k = 0; k < kFftSize / 2; k++) {
    g_cos[k] = (float)cos(2.0 * M_PI * k / kFftSize);
    g_sin[k] = (float)sin(2.0 * M_PI * k / kFftSize);
  }
  int bits = 0;
  while ((1 << bits) < kFftSize) bits++;
  for (int i = 0; i < kFftSize; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    g_bitrev[i] = (uint16_t)r;
  }

  // 42 edge frequencies equally spaced on the mel scale give 40 triangles.
  double edges[kNumMel + 2];
  double mlo = hz_to_mel(kFMin), mhi = hz_to_mel(kFMax);
  for (int i = 0; i < kNumMel + 2; i++)
    edges[i] = mel_to_hz(mlo + (mhi - mlo) * i / (kNumMel + 1));

  int offset = 0;
  for (int m = 0; m < kNumMel; m++) {
    double fl = edges[m], fc = edges[m + 1], fr = edges[m + 2];
    int start = -1, count = 0;
    g_mel_offset[m] = (uint16_t)offset;
    for (int k = 0; k < kNumBins; k++) {
      double f = (double)k * kSampleRate / kFftSize;
      double w = 0.0;
      if (f > fl && f < fc) w = (f - fl) / (fc - fl);
      else if (f >= fc && f < fr) w = (fr - f) / (fr - fc);
      if (w > 0.0) {
        if (start < 0) start = k;
        g_mel_weights[offset + count] = (float)w;
        count++;
      }
    }
    g_mel_start[m] = (uint16_t)(start < 0 ? 0 : start);
    g_mel_count[m] = (uint16_t)count;
    offset += count;
  }
}

void compute(const float* audio, float* out) {
  for (int f = 0; f < kNumFrames; f++) {
    const float* frame = audio + f * kHop;
    for (int n = 0; n < kFrameLen; n++) { g_re[n] = frame[n] * g_window[n]; g_im[n] = 0.0f; }
    for (int n = kFrameLen; n < kFftSize; n++) { g_re[n] = 0.0f; g_im[n] = 0.0f; }
    fft();
    for (int k = 0; k < kNumBins; k++) g_power[k] = g_re[k] * g_re[k] + g_im[k] * g_im[k];
    for (int m = 0; m < kNumMel; m++) {
      const float* w = g_mel_weights + g_mel_offset[m];
      const float* p = g_power + g_mel_start[m];
      float acc = 0.0f;
      for (int i = 0; i < g_mel_count[m]; i++) acc += w[i] * p[i];
      out[f * kNumMel + m] = logf(acc + kLogEps);
    }
  }
}

}  // namespace logmel
