
#include <Wire.h>
#include <Adafruit_AS7341.h>

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
