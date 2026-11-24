// SkyEar sensor node: pins and tuning. Values follow the Evaluation 2 design doc.
#pragma once

#define NODE_ID 1

// INMP441 I2S microphone (L/R tied to GND -> left channel)
#define PIN_I2S_SCK 4
#define PIN_I2S_WS  5
#define PIN_I2S_SD  6

// ESP-NOW radio link (built-in 2.4 GHz Wi-Fi radio, no extra module).
// ESP-NOW only works when both boards are on the same Wi-Fi channel, and the gateway
// must follow its router's channel. So the node scans for the router by name at boot
// and uses its channel. Only the network NAME is needed here, never the password.
#define ROUTER_SSID             "Mian"  // same network name as gateway/src/secrets.h
#define ESPNOW_FALLBACK_CHANNEL 1            // used if the router is not found

// BME280 (I2C)
#define PIN_I2C_SDA 8
#define PIN_I2C_SCL 9
#define BME280_ADDR 0x76

// Battery sense: 2 x 100k divider into GPIO1, so the ADC sees half the cell voltage
#define PIN_BATT_ADC   1
#define BATT_DIVIDER   2.0f
#define BATT_CAL       1.00f  // set once against a multimeter
#define BATTERY_SENSE_ENABLED 0  // 0 = powered from USB, no divider wired; 1 = 18650 + divider fitted

// Detection rules
#define ALARM_ON_P          0.80f  // a window "votes drone" at or above this
#define ALARM_ON_VOTES      3      // ... in this many of the last ...
#define ALARM_ON_WINDOW     4      // ... windows
#define ALARM_CLEAR_P       0.50f  // clear after P below this for ...
#define ALARM_CLEAR_WINDOWS 10     // ... this many consecutive windows (5 s at 0.5 s hop)
#define ALERT_REPEAT_MS     3000   // re-send ALERT while detection holds (broadcast has no ACK)

// Telemetry / heartbeat
#define HEARTBEAT_MS        5000  // gateway marks the node offline after 90 s of silence
#define HEARTBEAT_SLOW_MS   60000  // used below LOW_BATT_MV
#define LOW_BATT_MV         3400
#define CUTOFF_BATT_MV      3200   // final packet, then deep sleep
#define CUTOFF_SLEEP_MIN    30     // wake to re-check the cell

// Inference cadence: 1 s window, new decision every 0.5 s
#define HOP_SAMPLES 8000

// Hardware watchdog
#define WDT_TIMEOUT_S 10

// Uncomment to stream raw 16 kHz int16 audio over USB serial instead of detecting.
// Use tools/record_serial.py to save it as WAV for your own training data.
// #define RECORD_MODE
