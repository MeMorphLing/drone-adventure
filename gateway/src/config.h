// SkyEar gateway: pins and tuning. Values follow the Evaluation 2 design doc.
#pragma once
#include "secrets.h"  // copy secrets.example.h to secrets.h and fill it in

// ESP-NOW radio link. The gateway stays on its router's channel; while the router is
// unreachable it parks the radio on the last known channel so node packets keep arriving.
#define ESPNOW_FALLBACK_CHANNEL 1      // must match the node's fallback
#define WIFI_RETRY_MS           30000UL
#define WIFI_ATTEMPT_MS         8000UL

// SSD1306 OLED
#define PIN_I2C_SDA 21
#define PIN_I2C_SCL 22
#define OLED_ADDR   0x3C

// Actuators and inputs
// MINIMAL build (default): no external parts. The board's own blue LED is the alarm
// light and its BOOT button is the ACK button, so only the OLED is wired.
//   blue LED fast blink (5 Hz) = drone alarm, siren would be on
//   blue LED slow blink (1 Hz) = node offline
//   blue LED off               = normal
// Set EXTERNAL_ALARM_PARTS 1 once you add the relay, buzzer, LEDs and button.
#define EXTERNAL_ALARM_PARTS 0

#define PIN_STATUS_LED 2      // on-board blue LED on most ESP32 DevKit V1 boards
#if EXTERNAL_ALARM_PARTS
#define PIN_ACK_BUTTON 32     // external push button to GND, internal pull-up
#else
#define PIN_ACK_BUTTON 0      // on-board BOOT button (press it only after the board has booted)
#endif
#define PIN_RELAY      27
#define RELAY_ACTIVE   LOW    // most opto relay boards switch ON with a LOW input; check yours
#define PIN_BUZZER     25     // active buzzer through a 2N2222, HIGH = beep
#define PIN_LED_GREEN  4      // system online (external)
#define PIN_LED_AMBER  13     // node offline (external)

// Rules
#define SIREN_MAX_MS        60000UL  // automatic cut-off
#define BUZZER_PATTERN_MS   1600UL   // beep pattern after a new alert
#define NODE_OFFLINE_MS     90000UL  // 3 missed 30 s heartbeats
#define OVER_TEMP_C         60.0f
#define LOW_BATT_MV         3400

// ThingSpeak
#define TS_WRITE_PERIOD_MS  20000UL  // one combined write of all 8 fields
#define TS_MIN_GAP_MS       15500UL  // free-tier limit is 15 s; keep a margin
#define TS_QUEUE_LEN        200      // records kept while Wi-Fi is down (~66 min at 20 s)

// Time (for queued records' created_at)
#define NTP_SERVER "pool.ntp.org"

#define WDT_TIMEOUT_S 15
