// SkyEar sensor node: ESP32-S3 + INMP441 + SX1278 + BME280, battery powered.
// Listens, classifies 1 s windows every 0.5 s on-device, runs the 3-of-4 / 10-window
// hysteresis locally, and sends only classified events and telemetry over ESP-NOW.
#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <Adafruit_BME280.h>
#include <driver/i2s.h>
#include <esp_task_wdt.h>
#include <esp_sleep.h>
#include <math.h>

#include "config.h"
#include "logmel.h"
#include "model_data.h"
#include "skyear_packet.h"

#if SKYEAR_HAS_MODEL
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif

// ---------------------------------------------------------------- audio
static int16_t g_ring[logmel::kWindowSamples];  // last 1 s, oldest first after unroll
static int32_t g_i2s_buf[1024];
static float g_audio[logmel::kWindowSamples];
static float g_features[logmel::kNumFrames * logmel::kNumMel];

static void i2s_setup() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = logmel::kSampleRate;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;  // INMP441 sends 24 bits in a 32-bit slot
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 1024;  // 8 x 1024 samples = 512 ms of slack while we run inference
  cfg.use_apll = false;
  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = PIN_I2S_SCK;
  pins.ws_io_num = PIN_I2S_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = PIN_I2S_SD;
  i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr);
  i2s_set_pin(I2S_NUM_0, &pins);
  i2s_zero_dma_buffer(I2S_NUM_0);
}

// Reads exactly n samples and shifts them into the ring (drops the oldest n).
static void read_samples(int16_t* dst, int n) {
  int got = 0;
  while (got < n) {
    int want = min(n - got, (int)(sizeof(g_i2s_buf) / sizeof(g_i2s_buf[0])));
    size_t bytes = 0;
    i2s_read(I2S_NUM_0, g_i2s_buf, want * sizeof(int32_t), &bytes, portMAX_DELAY);
    int count = bytes / sizeof(int32_t);
    for (int i = 0; i < count; i++) dst[got + i] = (int16_t)(g_i2s_buf[i] >> 16);  // top 16 of 24 bits
    got += count;
  }
}

static void advance_ring() {
  memmove(g_ring, g_ring + HOP_SAMPLES, (logmel::kWindowSamples - HOP_SAMPLES) * sizeof(int16_t));
  read_samples(g_ring + logmel::kWindowSamples - HOP_SAMPLES, HOP_SAMPLES);
}

static float window_to_float_and_rms_dbfs() {
  double sum = 0.0;
  for (int i = 0; i < logmel::kWindowSamples; i++) {
    float s = g_ring[i] / 32768.0f;
    g_audio[i] = s;
    sum += (double)s * s;
  }
  double rms = sqrt(sum / logmel::kWindowSamples);
  return rms > 1e-6 ? 20.0f * log10f((float)rms) : -120.0f;
}

// ---------------------------------------------------------------- model
#if SKYEAR_HAS_MODEL
constexpr int kArenaSize = 100 * 1024;
static uint8_t* g_arena = nullptr;
static tflite::MicroInterpreter* g_interp = nullptr;
static TfLiteTensor* g_in = nullptr;
static TfLiteTensor* g_out = nullptr;

static bool model_setup() {
  static tflite::MicroErrorReporter reporter;
  const tflite::Model* model = tflite::GetModel(g_skyear_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("model schema %d != runtime %d\n", (int)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  g_arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!g_arena) { Serial.println("arena alloc failed"); return false; }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interp(model, resolver, g_arena, kArenaSize, &reporter);
  g_interp = &interp;
  if (g_interp->AllocateTensors() != kTfLiteOk) { Serial.println("AllocateTensors failed"); return false; }
  g_in = g_interp->input(0);
  g_out = g_interp->output(0);
  Serial.printf("model ok: %u bytes, arena used %u of %d\n", (unsigned)g_skyear_model_len,
                (unsigned)g_interp->arena_used_bytes(), kArenaSize);
  return true;
}

// Fills probs[SKYEAR_NUM_CLASSES]; returns inference time in microseconds.
static uint32_t model_run(float* probs) {
  const float in_scale = g_in->params.scale;
  const int in_zp = g_in->params.zero_point;
  int8_t* dst = g_in->data.int8;
  for (int i = 0; i < logmel::kNumFrames * logmel::kNumMel; i++) {
    int q = (int)lroundf(g_features[i] / in_scale) + in_zp;
    dst[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }
  uint32_t t0 = micros();
  g_interp->Invoke();
  uint32_t dt = micros() - t0;
  for (int c = 0; c < SKYEAR_NUM_CLASSES; c++)
    probs[c] = (g_out->data.int8[c] - g_out->params.zero_point) * g_out->params.scale;
  return dt;
}
#endif

// ---------------------------------------------------------------- sensors + radio
static Adafruit_BME280 g_bme;
static bool g_bme_ok = false;
static bool g_model_ok = false;
static uint16_t g_seq = 0;

static uint16_t battery_mv() {
#if !BATTERY_SENSE_ENABLED
  return 0;  // no cell fitted: 0 disables every low-battery rule below
#endif
  uint32_t acc = 0;
  for (int i = 0; i < 8; i++) acc += analogReadMilliVolts(PIN_BATT_ADC);
  return (uint16_t)(acc / 8.0f * BATT_DIVIDER * BATT_CAL);
}

static const uint8_t kBroadcast[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Channel of the router the gateway is connected to, so both boards meet on it.
static int find_router_channel() {
  int n = WiFi.scanNetworks(false, true);
  int ch = 0;
  for (int i = 0; i < n; i++) {
    if (WiFi.SSID(i) == ROUTER_SSID) { ch = WiFi.channel(i); break; }
  }
  WiFi.scanDelete();
  return ch;
}

static bool radio_setup() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  int ch = find_router_channel();
  if (ch <= 0) {
    ch = ESPNOW_FALLBACK_CHANNEL;
    Serial.printf("router \"%s\" not found, using channel %d\n", ROUTER_SSID, ch);
  }
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);
  if (esp_now_init() != ESP_OK) return false;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, kBroadcast, 6);
  peer.channel = 0;  // 0 = the channel set above
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) return false;
  Serial.printf("ESP-NOW ready on channel %d, node MAC %s\n", ch, WiFi.macAddress().c_str());
  return true;
}

struct WindowResult {
  float p_drone;
  float dbfs;
  uint8_t cls;
};

static void send_packet(SkyEarPacketType type, const WindowResult& w, float p_max, bool detection,
                        uint16_t batt) {
  SkyEarPacket p = {};
  p.node_id = NODE_ID;
  p.type = type;
  p.seq = g_seq++;
  p.p_drone_pct = (uint8_t)lroundf(constrain(w.p_drone, 0.0f, 1.0f) * 100);
  p.p_max_pct = (uint8_t)lroundf(constrain(p_max, 0.0f, 1.0f) * 100);
  p.cls = w.cls;
  p.sound_dbfs_x10 = (int16_t)lroundf(w.dbfs * 10);
  p.batt_mv = batt;
  if (detection) p.flags |= FLAG_DETECTION;
  if (batt > 1000 && batt < LOW_BATT_MV) p.flags |= FLAG_LOW_BATT;
  if (!g_model_ok) p.flags |= FLAG_NO_MODEL;
  if (g_bme_ok) {
    p.temp_x10 = (int16_t)lroundf(g_bme.readTemperature() * 10);
    p.hum_pct = (uint8_t)lroundf(constrain(g_bme.readHumidity(), 0.0f, 100.0f));
  } else {
    p.flags |= FLAG_BME_FAIL;
  }
  skyear_seal(&p);
  esp_err_t err = esp_now_send(kBroadcast, (const uint8_t*)&p, sizeof(p));  // ~1 ms, broadcast
  if (err != ESP_OK) Serial.printf("ESP-NOW send failed: %d\n", (int)err);
  Serial.printf("TX type=%u seq=%u p=%.2f pmax=%.2f det=%d batt=%umV\n", (unsigned)type,
                (unsigned)p.seq, w.p_drone, p_max, detection, (unsigned)batt);
}

// ---------------------------------------------------------------- detection state
static bool g_votes[ALARM_ON_WINDOW] = {};
static int g_vote_pos = 0;
static int g_clear_run = 0;
static bool g_detection = false;
static float g_p_max = 0.0f;
static uint32_t g_last_tx = 0;
static uint32_t g_last_alert = 0;

// Returns +1 on a rising edge (send ALERT), -1 on a falling edge (send CLEAR), 0 otherwise.
static int update_hysteresis(float p) {
  g_votes[g_vote_pos] = p >= ALARM_ON_P;
  g_vote_pos = (g_vote_pos + 1) % ALARM_ON_WINDOW;
  int votes = 0;
  for (bool v : g_votes) votes += v;
  g_clear_run = (p < ALARM_CLEAR_P) ? g_clear_run + 1 : 0;

  if (!g_detection && votes >= ALARM_ON_VOTES) {
    g_detection = true;
    g_clear_run = 0;
    return +1;
  }
  if (g_detection && g_clear_run >= ALARM_CLEAR_WINDOWS) {
    g_detection = false;
    return -1;
  }
  return 0;
}

// ---------------------------------------------------------------- setup / loop
void setup() {
  Serial.begin(921600);
  delay(300);
  esp_task_wdt_init(WDT_TIMEOUT_S, true);
  esp_task_wdt_add(nullptr);

  i2s_setup();
#ifndef RECORD_MODE
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_BATT_ADC, ADC_11db);
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  g_bme_ok = g_bme.begin(BME280_ADDR, &Wire);
  if (!g_bme_ok) Serial.println("BME280 not found");
  if (!radio_setup()) Serial.println("ESP-NOW init failed");
  logmel::init();
#if SKYEAR_HAS_MODEL
  g_model_ok = model_setup();
#endif
  // Prime the ring with a full second of audio.
  read_samples(g_ring, logmel::kWindowSamples);
  Serial.printf("SkyEar node %d ready, model=%s\n", NODE_ID, g_model_ok ? "yes" : "NO (placeholder)");
#endif
}

#ifdef RECORD_MODE
// Frames of [0xA5 0x5A] + 512 little-endian int16 samples at 16 kHz, read by
// tools/record_serial.py. The sync bytes let the script lock on mid-stream.
// Same I2S path and scaling as detection, so recordings match what the model sees.
void loop() {
  static int16_t chunk[512];
  static const uint8_t sync[2] = {0xA5, 0x5A};
  read_samples(chunk, 512);
  Serial.write(sync, 2);
  Serial.write((const uint8_t*)chunk, sizeof(chunk));
  esp_task_wdt_reset();
}
#else
void loop() {
  advance_ring();  // blocks for ~0.5 s of new audio
  esp_task_wdt_reset();

  WindowResult w = {};
  w.dbfs = window_to_float_and_rms_dbfs();
  uint32_t t0 = micros();
  logmel::compute(g_audio, g_features);
  uint32_t t_feat = micros() - t0, t_inf = 0;

  float probs[SKYEAR_NUM_CLASSES] = {};
#if SKYEAR_HAS_MODEL
  if (g_model_ok) t_inf = model_run(probs);
#endif
  int top = 0;
  for (int c = 1; c < SKYEAR_NUM_CLASSES; c++) if (probs[c] > probs[top]) top = c;
  w.p_drone = probs[SKYEAR_DRONE_INDEX];
  w.cls = (uint8_t)top;
  g_p_max = max(g_p_max, w.p_drone);

  int edge = update_hysteresis(w.p_drone);
  Serial.printf("win %s p=%.2f rms=%.1fdBFS feat=%luus inf=%luus det=%d\n", kSkyEarClassNames[top],
                w.p_drone, w.dbfs, (unsigned long)t_feat, (unsigned long)t_inf, g_detection);

  uint32_t now = millis();
  uint16_t batt = battery_mv();

  if (batt > 1000 && batt < CUTOFF_BATT_MV) {  // > 1 V guards against an unplugged sense line
    send_packet(PKT_LOWBATT_FINAL, w, g_p_max, g_detection, batt);
    esp_now_deinit();
    WiFi.mode(WIFI_OFF);
    esp_sleep_enable_timer_wakeup((uint64_t)CUTOFF_SLEEP_MIN * 60ULL * 1000000ULL);
    esp_deep_sleep_start();
  }

  if (edge > 0 || (g_detection && now - g_last_alert >= ALERT_REPEAT_MS)) {
    send_packet(PKT_ALERT, w, g_p_max, true, batt);
    g_last_alert = g_last_tx = now;
    g_p_max = 0.0f;
  } else if (edge < 0) {
    send_packet(PKT_CLEAR, w, g_p_max, false, batt);
    g_last_tx = now;
    g_p_max = 0.0f;
  } else {
    uint32_t period = (batt > 1000 && batt < LOW_BATT_MV) ? HEARTBEAT_SLOW_MS : HEARTBEAT_MS;
    if (now - g_last_tx >= period) {
      send_packet(PKT_TELEMETRY, w, g_p_max, g_detection, batt);
      g_last_tx = now;
      g_p_max = 0.0f;
    }
  }
}
#endif
