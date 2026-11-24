// SkyEar radio packet format (sent over ESP-NOW), shared by the sensor node and the gateway.
// Fixed 19-byte binary frame, little-endian, CRC-8 over every byte before `crc`.
#pragma once
#include <stdint.h>
#include <stddef.h>

#define SKYEAR_MAGIC   0x5E
#define SKYEAR_VERSION 1

enum SkyEarPacketType : uint8_t {
  PKT_TELEMETRY     = 1,  // periodic status, doubles as the heartbeat
  PKT_ALERT         = 2,  // drone confirmed (3 of last 4 windows >= 0.8)
  PKT_CLEAR         = 3,  // P(drone) < 0.5 for 10 consecutive windows
  PKT_LOWBATT_FINAL = 4,  // battery < 3.2 V, node goes to deep sleep after this
};

enum SkyEarFlags : uint8_t {
  FLAG_DETECTION = 0x01,  // node's hysteresis state is "drone"
  FLAG_LOW_BATT  = 0x02,  // battery < 3.4 V
  FLAG_NO_MODEL  = 0x04,  // firmware built without a trained model
  FLAG_BME_FAIL  = 0x08,  // BME280 not responding
};

#pragma pack(push, 1)
struct SkyEarPacket {
  uint8_t  magic;           // SKYEAR_MAGIC
  uint8_t  version;         // SKYEAR_VERSION
  uint8_t  node_id;         // 1..255
  uint8_t  type;            // SkyEarPacketType
  uint16_t seq;             // increments every packet
  uint8_t  p_drone_pct;     // latest P(drone) x 100
  uint8_t  p_max_pct;       // max P(drone) x 100 since the last packet
  uint8_t  cls;             // index of the top class in the latest window
  uint8_t  flags;           // SkyEarFlags
  int16_t  sound_dbfs_x10;  // RMS level of the latest 1 s window, dBFS x 10
  int16_t  temp_x10;        // BME280 temperature, deg C x 10
  uint8_t  hum_pct;         // BME280 relative humidity, %
  uint8_t  reserved;
  uint16_t batt_mv;         // battery voltage, mV
  uint8_t  crc;             // CRC-8 (poly 0x07) over bytes 0..17
};
#pragma pack(pop)

static_assert(sizeof(SkyEarPacket) == 19, "SkyEarPacket must be 19 bytes");

static inline uint8_t skyear_crc8(const uint8_t* data, size_t len) {
  uint8_t crc = 0x00;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int b = 0; b < 8; b++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  }
  return crc;
}

static inline void skyear_seal(SkyEarPacket* p) {
  p->magic = SKYEAR_MAGIC;
  p->version = SKYEAR_VERSION;
  p->crc = skyear_crc8((const uint8_t*)p, sizeof(SkyEarPacket) - 1);
}

static inline bool skyear_valid(const SkyEarPacket* p) {
  return p->magic == SKYEAR_MAGIC && p->version == SKYEAR_VERSION &&
         p->crc == skyear_crc8((const uint8_t*)p, sizeof(SkyEarPacket) - 1);
}
