/*  ============================================================================
 *  AS7341 PPFD / PAR MEASUREMENT FIRMWARE  (ESP32 + AS7341 breakout)
 *  ----------------------------------------------------------------------------
 *  Purpose
 *    Reads all spectral channels of the AS7341 and returns averaged, low-noise
 *    measurements over USB serial as a single CSV line per request. The companion
 *    Python script (../capture/capture_calibration.py) drives this firmware, adds
 *    the experiment metadata, and writes an analysis-ready CSV file
 *    (analysed with ../analysis/analyze_calibration.R).
 *
 *  Why this design (scientific rationale)
 *    - Acquisition settings (gain, integration time) are FIXED and documented.
 *      Both Baumker et al. (2021, Sensors 21:3390) and Rahman, Jamil & Pearce
 *      (2025, Electronics 14:2225) keep these constant so a single calibration
 *      stays valid. Do NOT change them once you have calibrated.
 *    - Each measurement is the mean of BURST samples; the standard deviation is
 *      also reported, giving you per-point measurement precision for the paper.
 *    - RAW counts of 10 channels are reported (8 visible F1-F8 + Clear + NIR);
 *      the AS7341 flicker-detection channel is not used. Baumker et al. found
 *      the NIR channel can improve the PPFD regression, so it is logged too.
 *      Derived quantities (basic counts, calibrated PPFD) are computed later
 *      in R from these raw data (store raw, derive in analysis).
 *    - The standard deviation reported for each channel is the population SD
 *      of the BURST samples.
 *
 *  Hardware / wiring (I2C)
 *      ESP32 3V3   -> AS7341 VIN     (3.3 V only, never 5 V)
 *      ESP32 GND   -> AS7341 GND
 *      ESP32 GPIO21-> AS7341 SDA
 *      ESP32 GPIO22-> AS7341 SCL
 *
 *  Library required (Arduino IDE -> Library Manager)
 *      "Adafruit AS7341" (installs Adafruit BusIO as a dependency)
 *
 *  Serial protocol (115200 baud)
 *      Host sends   ->   Firmware replies
 *      "PING"            "PONG"
 *      "INFO"            "INFO,gain=..,atime=..,astep=..,intMs=.."
 *      "READ"            "DATA,<settings>,<10 means>,<10 sds>"   (one line)
 *      (on error)        "ERROR: <message>"
 *  ============================================================================
 */

#include <Wire.h>
#include <Adafruit_AS7341.h>

// ----------------------------- CONFIGURATION --------------------------------
// Acquisition settings. Keep FIXED after calibration.
// Integration time t_int = (ATIME+1) * (ASTEP+1) * 2.78 us.
// 35, 999 -> 100.08 ms  (matches Rahman et al. 2025).
// Settings used for all data in ../data/AS7341_calibration_data.csv:
// gain 1x, ATIME 35, ASTEP 999, BURST 10.
const uint8_t        ATIME = 35;
const uint16_t       ASTEP = 999;
const as7341_gain_t  GAIN  = AS7341_GAIN_1X;   // gain = 1 (good against saturation)
const int            BURST = 10;               // samples averaged per measurement

const uint8_t SDA_PIN = 21;
const uint8_t SCL_PIN = 22;
// ----------------------------------------------------------------------------

Adafruit_AS7341 as7341;

// Channel order as returned by readAllChannels(), skipping the duplicate
// Clear/NIR at buffer indices 4 and 5.
const char* CH_NAMES[10] =
  {"F1_415","F2_445","F3_480","F4_515","F5_555","F6_590","F7_630","F8_680","Clear","NIR"};
const int   CH_IDX[10]   = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11};

float integrationTimeMs() {
  return (float)(ATIME + 1) * (float)(ASTEP + 1) * 2.78f / 1000.0f;
}

float gainValue() {
  switch (GAIN) {
    case AS7341_GAIN_0_5X: return 0.5f;
    case AS7341_GAIN_1X:   return 1.0f;
    case AS7341_GAIN_2X:   return 2.0f;
    case AS7341_GAIN_4X:   return 4.0f;
    case AS7341_GAIN_8X:   return 8.0f;
    case AS7341_GAIN_16X:  return 16.0f;
    case AS7341_GAIN_32X:  return 32.0f;
    case AS7341_GAIN_64X:  return 64.0f;
    case AS7341_GAIN_128X: return 128.0f;
    case AS7341_GAIN_256X: return 256.0f;
    case AS7341_GAIN_512X: return 512.0f;
    default:               return 1.0f;
  }
}

void printInfo() {
  Serial.print("INFO,gain=");  Serial.print(gainValue(), 1);
  Serial.print(",atime=");     Serial.print(ATIME);
  Serial.print(",astep=");     Serial.print(ASTEP);
  Serial.print(",intMs=");     Serial.println(integrationTimeMs(), 3);
}

void takeMeasurement() {
  uint16_t readings[12];
  double   sum[10]   = {0};
  double   sumsq[10] = {0};
  const double fullScale = (double)(ATIME + 1) * (double)(ASTEP + 1); // max count
  bool     saturated = false;

  for (int n = 0; n < BURST; n++) {
    if (!as7341.readAllChannels(readings)) {
      Serial.println("ERROR: readAllChannels failed");
      return;
    }
    for (int c = 0; c < 10; c++) {
      double v = (double)readings[CH_IDX[c]];
      sum[c]   += v;
      sumsq[c] += v * v;
      if (v >= 0.95 * fullScale) saturated = true;
    }
    delay(5);
  }

  // CSV: DATA,gain,atime,astep,intMs,burst,saturated, mean x10, sd x10
  Serial.print("DATA");
  Serial.print(",");  Serial.print(gainValue(), 1);
  Serial.print(",");  Serial.print(ATIME);
  Serial.print(",");  Serial.print(ASTEP);
  Serial.print(",");  Serial.print(integrationTimeMs(), 3);
  Serial.print(",");  Serial.print(BURST);
  Serial.print(",");  Serial.print(saturated ? 1 : 0);

  for (int c = 0; c < 10; c++) {                 // means
    double mean = sum[c] / BURST;
    Serial.print(",");  Serial.print(mean, 2);
  }
  for (int c = 0; c < 10; c++) {                 // standard deviations
    double mean = sum[c] / BURST;
    double var  = (sumsq[c] / BURST) - (mean * mean);
    if (var < 0) var = 0;                         // guard tiny negative rounding
    Serial.print(",");  Serial.print(sqrt(var), 2);
  }
  Serial.println();
}

String cmd = "";

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!as7341.begin()) {
    Serial.println("ERROR: AS7341 not found. Check wiring: VIN->3V3, GND->GND, SDA->21, SCL->22.");
    while (true) delay(1000);
  }
  as7341.setATIME(ATIME);
  as7341.setASTEP(ASTEP);
  as7341.setGain(GAIN);

  Serial.println("READY");   // capture script waits for this
}

void loop() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      cmd.trim();
      if      (cmd.equalsIgnoreCase("READ")) takeMeasurement();
      else if (cmd.equalsIgnoreCase("PING")) Serial.println("PONG");
      else if (cmd.equalsIgnoreCase("INFO")) printInfo();
      else if (cmd.length() > 0)             Serial.println("ERROR: unknown command");
      cmd = "";
    } else {
      cmd += ch;
      if (cmd.length() > 32) cmd = "";          // overflow guard
    }
  }
}
