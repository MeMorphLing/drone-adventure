// SkyEar gateway: ESP32 DevKit V1 in the guard room.
// Receives ESP-NOW events, drives the siren / buzzer / OLED / LEDs locally, and writes
// one combined 8-field record to ThingSpeak every 20 s. No inference happens here,
// and nothing on the alarm path depends on Wi-Fi.
#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <esp_task_wdt.h>
#include <time.h>

#include "config.h"
#include "skyear_packet.h"

static const char* const kClassNames[] = {"Ambient", "Drone", "Speech", "Vehicle"};

static Adafruit_SSD1306 oled(128, 64, &Wire, -1);

// ---------------------------------------------------------------- state
struct NodeState {
  bool seen = false;
  uint32_t last_rx_ms = 0;
  SkyEarPacket last = {};
  int rssi = 0;
  float p_max = 0.0f;        // max P(drone) since the last cloud write
  float sound_dbfs = 0.0f;
  float temp_c = NAN;
  float hum = NAN;
  float batt_v = NAN;
};
static NodeState g_node;

static bool g_detection = false;    // Field 2
static bool g_siren = false;        // Field 8
static uint32_t g_siren_start = 0;
static bool g_oled_latched = false; // alert screen stays until acknowledged
static uint8_t g_latched_pct = 0;
static uint32_t g_buzz_start = 0;
static bool g_buzzing = false;
static bool g_force_write = false;  // write as soon as the 15 s gap allows
static uint32_t g_last_write = 0;
static bool g_offline = false;
static bool g_ack_after_detection = false;

// ---------------------------------------------------------------- outputs
static void relay(bool on) {
#if EXTERNAL_ALARM_PARTS
  digitalWrite(PIN_RELAY, on ? RELAY_ACTIVE : !RELAY_ACTIVE);
#else
  (void)on;  // no relay fitted: the "siren" is the on-board LED + flashing OLED
#endif
}

static void siren_on() {
  if (!g_siren) {
    g_siren = true;
    g_siren_start = millis();
    relay(true);
    Serial.println(EXTERNAL_ALARM_PARTS ? "ALARM: relay ON (GPIO27)" : "ALARM: on-board LED + OLED alert");
  }
}

static void siren_off(const char* why) {
  if (g_siren) {
    g_siren = false;
    relay(false);
    Serial.printf("relay OFF (%s)\n", why);
  }
}

static void buzz_start() {
  g_buzzing = true;
  g_buzz_start = millis();
}

static void buzzer_tick() {
#if !EXTERNAL_ALARM_PARTS
  if (g_buzzing && millis() - g_buzz_start >= BUZZER_PATTERN_MS) g_buzzing = false;
  return;
#endif
  if (!g_buzzing) { digitalWrite(PIN_BUZZER, LOW); return; }
  uint32_t t = millis() - g_buzz_start;
  if (t >= BUZZER_PATTERN_MS) { g_buzzing = false; digitalWrite(PIN_BUZZER, LOW); return; }
  digitalWrite(PIN_BUZZER, ((t / 200) % 2) == 0 ? HIGH : LOW);  // 200 ms on / off
}

static const char* status_text() {
  if (g_offline) return "NODE OFFLINE";
  if (g_node.seen && (g_node.last.flags & FLAG_LOW_BATT)) return "LOW BATTERY";
  if (g_node.seen && !isnan(g_node.temp_c) && g_node.temp_c > OVER_TEMP_C) return "OVER TEMP";
  if (g_node.seen && (g_node.last.flags & FLAG_NO_MODEL)) return "NO MODEL";
  return "OK";
}

// ---------------------------------------------------------------- OLED
static size_t queue_count();

static void draw_oled() {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  if (g_oled_latched) {
    bool inv = g_siren && ((millis() / 400) % 2 == 0);
    oled.invertDisplay(inv);
    oled.setTextSize(3); oled.setCursor(19, 0);  oled.print("DRONE");
    oled.setTextSize(2); oled.setCursor(28, 28); oled.print("NODE 1");
    oled.setTextSize(g_siren ? 2 : 1);
    if (g_siren) { oled.setCursor(46, 47); oled.printf("%u%%", g_latched_pct); }
    else { oled.setCursor(19, 50); oled.print("ACK - SIREN OFF"); }
    oled.display();
    return;
  }
  oled.invertDisplay(false);
  if (g_offline) {
    oled.setTextSize(2); oled.setCursor(28, 2);  oled.print("NODE 1");
    oled.setCursor(22, 24); oled.print("OFFLINE");
    oled.setTextSize(1); oled.setCursor(22, 48); oled.print("no packet 90 s");
    oled.display();
    return;
  }
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.printf("SkyEar GW    %s", WiFi.isConnected() ? "WiFi OK" : "NO WiFi");
  oled.drawFastHLine(0, 9, 128, SSD1306_WHITE);
  oled.setCursor(0, 13);
  if (!g_node.seen) { oled.print("waiting for node..."); oled.display(); return; }
  oled.print("NODE 1  ONLINE");
  oled.setCursor(0, 24);
  uint8_t c = g_node.last.cls;
  oled.printf("Class: %s", c < 4 ? kClassNames[c] : "?");
  oled.setCursor(0, 35);
  oled.printf("P(drone): %.2f", g_node.last.p_drone_pct / 100.0f);
  oled.setCursor(0, 46);
  oled.printf("Sound %.0fdB RSSI %d", g_node.sound_dbfs, g_node.rssi);
  oled.setCursor(0, 57);
  size_t q = queue_count();
  if (q) oled.printf("TS queue: %u", (unsigned)q);
  else oled.print(status_text());
  oled.display();
}

// ---------------------------------------------------------------- radio RX
static void handle_packet(const SkyEarPacket& p, int rssi) {
  g_node.seen = true;
  g_node.last_rx_ms = millis();
  g_node.last = p;
  g_node.rssi = rssi;
  g_node.p_max = max(g_node.p_max, max(p.p_drone_pct, p.p_max_pct) / 100.0f);
  g_node.sound_dbfs = p.sound_dbfs_x10 / 10.0f;
  g_node.batt_v = p.batt_mv / 1000.0f;
  if (!(p.flags & FLAG_BME_FAIL)) {
    g_node.temp_c = p.temp_x10 / 10.0f;
    g_node.hum = p.hum_pct;
  }
  if (g_offline) { g_offline = false; Serial.println("node 1 back online"); }

  Serial.printf("RX node=%u type=%u seq=%u p=%.2f rssi=%d\n", p.node_id, p.type, p.seq,
                p.p_drone_pct / 100.0f, rssi);

  switch (p.type) {
    case PKT_ALERT:
      if (!g_detection) {  // rising edge: repeats only refresh state
        g_detection = true;
        g_oled_latched = true;
        g_ack_after_detection = false;
        g_latched_pct = p.p_drone_pct;
        siren_on();
        buzz_start();
        g_force_write = true;
      }
      break;
    case PKT_CLEAR:
      if (g_detection) {
        g_detection = false;  // siren keeps its own ACK / 60 s rules
        g_force_write = true;
        Serial.println("detection state -> 0");
      }
      break;
    case PKT_LOWBATT_FINAL:
      Serial.println("node sent its final low-battery packet");
      g_force_write = true;
      break;
    default:
      break;
  }
}

// The ESP-NOW callback runs in the Wi-Fi task, so it only copies the frame into a
// queue; all the alarm logic runs in loop().
struct RxItem {
  SkyEarPacket p;
  int len;
};
static QueueHandle_t g_rx_q = nullptr;
static volatile int g_last_rssi = 0;

static void on_espnow_recv(const uint8_t* mac, const uint8_t* data, int len) {
  (void)mac;
  RxItem it = {};
  it.len = len;
  memcpy(&it.p, data, len < (int)sizeof(SkyEarPacket) ? len : sizeof(SkyEarPacket));
  xQueueSend(g_rx_q, &it, 0);
}

// ESP-NOW frames are vendor-specific action frames (subtype 0xD0, category 127).
// The receive callback does not report signal strength, so read it here.
static void on_promisc(void* buf, wifi_promiscuous_pkt_type_t type) {
  if (type != WIFI_PKT_MGMT) return;
  const wifi_promiscuous_pkt_t* pkt = (const wifi_promiscuous_pkt_t*)buf;
  if (pkt->rx_ctrl.sig_len < 25) return;
  const uint8_t* f = pkt->payload;
  if (f[0] == 0xD0 && f[24] == 127) g_last_rssi = pkt->rx_ctrl.rssi;
}

static void radio_poll() {
  RxItem it;
  while (xQueueReceive(g_rx_q, &it, 0) == pdTRUE) {
    if (it.len != (int)sizeof(SkyEarPacket) || !skyear_valid(&it.p)) {
      Serial.printf("RX ignored: %d bytes or bad CRC\n", it.len);
      continue;
    }
    handle_packet(it.p, g_last_rssi);
  }
}

// ---------------------------------------------------------------- Wi-Fi + channel
static int g_wifi_ch = ESPNOW_FALLBACK_CHANNEL;
static uint8_t g_bssid[6];
static bool g_have_bssid = false;
static uint32_t g_wifi_try = 0;
static bool g_wifi_trying = false;

static void wifi_find_router() {
  int n = WiFi.scanNetworks(false, true);
  for (int i = 0; i < n; i++) {
    if (WiFi.SSID(i) == WIFI_SSID) {
      g_wifi_ch = WiFi.channel(i);
      memcpy(g_bssid, WiFi.BSSID(i), 6);
      g_have_bssid = true;
      break;
    }
  }
  WiFi.scanDelete();
  Serial.printf("router %s, channel %d\n", g_have_bssid ? "found" : "NOT found", g_wifi_ch);
}

static void wifi_connect() {
  // Giving the channel and BSSID connects without scanning other channels, so ESP-NOW
  // reception is not interrupted by a channel sweep.
  if (g_have_bssid) WiFi.begin(WIFI_SSID, WIFI_PASSWORD, g_wifi_ch, g_bssid);
  else WiFi.begin(WIFI_SSID, WIFI_PASSWORD, g_wifi_ch);
  g_wifi_try = millis();
  g_wifi_trying = true;
}

static void wifi_tick() {
  if (WiFi.isConnected()) { g_wifi_trying = false; return; }
  uint32_t now = millis();
  if (g_wifi_trying && now - g_wifi_try >= WIFI_ATTEMPT_MS) {
    // Attempt failed: stop it and park the radio back on the node's channel.
    WiFi.disconnect(false);
    esp_wifi_set_channel(g_wifi_ch, WIFI_SECOND_CHAN_NONE);
    g_wifi_trying = false;
  }
  if (!g_wifi_trying && now - g_wifi_try >= WIFI_RETRY_MS) wifi_connect();
}

// ---------------------------------------------------------------- ThingSpeak
struct Record {
  time_t ts;  // 0 if NTP was not synced yet
  float f1, f3, f4, f5, f6;
  int8_t f2, f8;
  int16_t f7;
  char status[16];
};
// Backlog of records not yet accepted by ThingSpeak. Owned by the cloud task;
// the main loop only reads g_q_count for the OLED.
static Record g_queue[TS_QUEUE_LEN];
static size_t g_q_head = 0;
static volatile size_t g_q_count = 0;
static QueueHandle_t g_cloud_q = nullptr;  // main loop -> cloud task

static size_t queue_count() { return g_q_count; }

static void queue_push(const Record& r) {
  size_t idx = (g_q_head + g_q_count) % TS_QUEUE_LEN;
  if (g_q_count == TS_QUEUE_LEN) g_q_head = (g_q_head + 1) % TS_QUEUE_LEN;  // drop oldest
  else g_q_count++;
  g_queue[idx] = r;
}

static Record snapshot() {
  Record r = {};
  r.ts = time(nullptr) > 1700000000 ? time(nullptr) : 0;
  r.f1 = g_node.p_max;
  r.f2 = g_detection ? 1 : 0;
  r.f3 = g_node.sound_dbfs;
  r.f4 = g_node.temp_c;
  r.f5 = g_node.hum;
  r.f6 = g_node.batt_v;
  r.f7 = (int16_t)g_node.rssi;
  r.f8 = g_siren ? 1 : 0;
  strncpy(r.status, status_text(), sizeof(r.status) - 1);
  return r;
}

static String fields_kv(const Record& r, bool json) {
  // Writes fields that have a value; NaN (no node data yet) is left out.
  String s;
  auto add = [&](const char* k, const String& v) {
    if (json) s += (s.length() ? ",\"" : "\"") + String(k) + "\":\"" + v + "\"";
    else s += "&" + String(k) + "=" + v;
  };
  if (g_node.seen) {
    add("field1", String(r.f1, 2));
    add("field3", String(r.f3, 1));
    if (!isnan(r.f4)) add("field4", String(r.f4, 1));
    if (!isnan(r.f5)) add("field5", String(r.f5, 0));
    if (!isnan(r.f6)) add("field6", String(r.f6, 2));
    add("field7", String(r.f7));
  }
  add("field2", String(r.f2));
  add("field8", String(r.f8));
  add("status", String(r.status));
  return s;
}

static String iso_utc(time_t t) {
  struct tm tm;
  gmtime_r(&t, &tm);
  char buf[24];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return String(buf);
}

static bool http_post(const String& url, const String& body, const char* type) {
  WiFiClient client;
    // TODO: pin the api.thingspeak.com root CA for production
  HTTPClient http;
  if (!http.begin(client, url)) return false;
  http.addHeader("Content-Type", type);
  http.setTimeout(8000);
  int code = http.POST(body);
  String resp = http.getString();
  http.end();
  Serial.printf("ThingSpeak POST -> %d %s\n", code, resp.substring(0, 40).c_str());
  if (code < 0) {
    IPAddress ip; bool dns = WiFi.hostByName("api.thingspeak.com", ip);
    Serial.printf("  diag: myIP=%s gw=%s dns=%s resolve=%s %s\n", WiFi.localIP().toString().c_str(), WiFi.gatewayIP().toString().c_str(), WiFi.dnsIP().toString().c_str(), dns ? "ok" : "FAIL", ip.toString().c_str());
  }
  // /update returns the new entry id ("0" means rejected); bulk returns {"success":true}
  return code == 200 && resp != "0";
}

static bool ts_write_single(const Record& r) {
  String body = String("api_key=") + TS_WRITE_KEY + fields_kv(r, false);
  body.replace(" ", "+");
  return http_post("http://api.thingspeak.com/update", body, "application/x-www-form-urlencoded");
}

// Sends every queued record plus the current one in a single bulk call.
static bool ts_write_bulk(const Record& now_rec) {
  queue_push(now_rec);
  String body = String("{\"write_api_key\":\"") + TS_WRITE_KEY + "\",\"updates\":[";
  for (size_t i = 0; i < g_q_count; i++) {
    const Record& r = g_queue[(g_q_head + i) % TS_QUEUE_LEN];
    if (i) body += ",";
    body += "{";
    if (r.ts) body += "\"created_at\":\"" + iso_utc(r.ts) + "\",";
    else body += "\"delta_t\":0,";
    body += fields_kv(r, true) + "}";
  }
  body += "]}";
  bool ok = http_post(String("http://api.thingspeak.com/channels/") + TS_CHANNEL_ID + "/bulk_update.json",
                      body, "application/json");
  if (ok) { Serial.printf("uploaded %u queued records\n", (unsigned)g_q_count); g_q_count = 0; g_q_head = 0; }
  return ok;
}

// Runs on core 0 so a slow or failing HTTPS call can never delay the alarm path on core 1.
static void cloud_task(void*) {
  Record r;
  for (;;) {
    if (xQueueReceive(g_cloud_q, &r, portMAX_DELAY) != pdTRUE) continue;
    if (!WiFi.isConnected()) {
      queue_push(r);
      Serial.printf("WiFi down: record queued (%u in queue)\n", (unsigned)g_q_count);
      continue;
    }
    bool ok = g_q_count ? ts_write_bulk(r) : ts_write_single(r);
    if (!ok && !g_q_count) queue_push(r);  // keep it for the next bulk attempt
  }
}

// Called from the main loop: decides when a write is due and hands a snapshot over.
static void cloud_tick() {
  uint32_t now = millis();
  uint32_t gap = now - g_last_write;
  bool due = gap >= TS_WRITE_PERIOD_MS || (g_force_write && gap >= TS_MIN_GAP_MS);
  if (!due) return;
  g_last_write = now;
  g_force_write = false;
  Record r = snapshot();
  g_node.p_max = 0.0f;  // Field 1 is the max over each interval
  if (xQueueSend(g_cloud_q, &r, 0) != pdTRUE) Serial.println("cloud task busy, record dropped");
}

// ---------------------------------------------------------------- inputs + timers
static void ack_poll() {
  static bool last = HIGH;
  static uint32_t changed = 0;
  bool level = digitalRead(PIN_ACK_BUTTON);
  if (level != last && millis() - changed > 40) {
    changed = millis();
    last = level;
    if (level == LOW) {  // pressed
      Serial.println("ACK button (GPIO32)");
      siren_off("acknowledged");
      g_buzzing = false;
      g_ack_after_detection = true;  // screen unlatches once detection has also cleared
      g_force_write = true;
    }
  }
}

static void safety_tick() {
  uint32_t now = millis();
  if (g_siren && now - g_siren_start >= SIREN_MAX_MS) siren_off("60 s cut-off");
  // Latched screen clears once the node has cleared and the guard has acknowledged.
  if (g_oled_latched && !g_siren && !g_detection && g_ack_after_detection) {
    g_oled_latched = false;
    g_ack_after_detection = false;
  }
  bool offline = g_node.seen && now - g_node.last_rx_ms >= NODE_OFFLINE_MS;
  if (offline && !g_offline) {
    g_offline = true;
    g_force_write = true;
    Serial.println("node 1: no packet for 90 s -> NODE OFFLINE");
  }
#if EXTERNAL_ALARM_PARTS
  digitalWrite(PIN_LED_AMBER, g_offline ? HIGH : LOW);
  digitalWrite(PIN_LED_GREEN, (g_node.seen && !g_offline) ? HIGH : LOW);
#endif
  // On-board LED: fast blink = alarm, slow blink = node offline, off = normal.
  bool led = false;
  if (g_siren) led = (now / 100) % 2 == 0;
  else if (g_offline) led = (now / 500) % 2 == 0;
  digitalWrite(PIN_STATUS_LED, led ? HIGH : LOW);
}

// ---------------------------------------------------------------- setup / loop
void setup() {
  // Relay de-energised before the pin becomes an output, so boot never fires the siren.
  pinMode(PIN_STATUS_LED, OUTPUT); digitalWrite(PIN_STATUS_LED, LOW);
#if EXTERNAL_ALARM_PARTS
  digitalWrite(PIN_RELAY, !RELAY_ACTIVE);
  pinMode(PIN_RELAY, OUTPUT);
  relay(false);
  pinMode(PIN_BUZZER, OUTPUT);     digitalWrite(PIN_BUZZER, LOW);
  pinMode(PIN_LED_GREEN, OUTPUT);  digitalWrite(PIN_LED_GREEN, LOW);
  pinMode(PIN_LED_AMBER, OUTPUT);  digitalWrite(PIN_LED_AMBER, LOW);
#endif
  pinMode(PIN_ACK_BUTTON, INPUT_PULLUP);

  Serial.begin(115200);
  delay(200);
  esp_task_wdt_init(WDT_TIMEOUT_S, true);
  esp_task_wdt_add(nullptr);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) Serial.println("OLED not found");
  oled.clearDisplay(); oled.setTextSize(1); oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 24); oled.print("SkyEar gateway boot"); oled.display();

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(false);  // reconnects are managed in wifi_tick() to keep the channel
  wifi_find_router();
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(g_wifi_ch, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  g_rx_q = xQueueCreate(8, sizeof(RxItem));
  if (esp_now_init() != ESP_OK) Serial.println("ESP-NOW init failed");
  esp_now_register_recv_cb(on_espnow_recv);
  wifi_promiscuous_filter_t filt = {};
  filt.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT;
  esp_wifi_set_promiscuous_filter(&filt);
  esp_wifi_set_promiscuous_rx_cb(on_promisc);
  esp_wifi_set_promiscuous(true);

  wifi_connect();  // non-blocking; the alarm path never waits on it
  configTime(0, 0, NTP_SERVER);
  Serial.printf("gateway MAC %s\n", WiFi.macAddress().c_str());

  g_cloud_q = xQueueCreate(4, sizeof(Record));
  xTaskCreatePinnedToCore(cloud_task, "cloud", 8192, nullptr, 1, nullptr, 0);
  g_last_write = millis();
  Serial.println("SkyEar gateway ready");
}

void loop() {
  esp_task_wdt_reset();
  radio_poll();
  wifi_tick();
  ack_poll();
  safety_tick();
  buzzer_tick();
  static uint32_t last_draw = 0;
  if (millis() - last_draw >= 200) { last_draw = millis(); draw_oled(); }
  cloud_tick();  // never blocks: the upload itself runs in cloud_task
}
