// Host-side check that the firmware front end matches training/features.py.
// Build: g++ -O2 -std=c++17 -I../node/src test_logmel.cpp ../node/src/logmel.cpp -o test_logmel
// Usage: ./test_logmel in.f32 out.f32   (16000 float32 samples in, 49*40 float32 out)
#include <cstdio>
#include <vector>
#include "logmel.h"

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s in.f32 out.f32\n", argv[0]); return 2; }
  std::vector<float> audio(logmel::kWindowSamples), out(logmel::kNumFrames * logmel::kNumMel);
  FILE* f = fopen(argv[1], "rb");
  if (!f || fread(audio.data(), sizeof(float), audio.size(), f) != audio.size()) { fprintf(stderr, "read failed\n"); return 1; }
  fclose(f);
  logmel::init();
  logmel::compute(audio.data(), out.data());
  f = fopen(argv[2], "wb");
  fwrite(out.data(), sizeof(float), out.size(), f);
  fclose(f);
  return 0;
}
