#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <DHT.h>
#include <RTClib.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2561_U.h>

#define DHTPIN 2
#define DHTTYPE DHT11
#define SOIL_PIN A0
#define SD_CS 10

DHT dht(DHTPIN, DHTTYPE);
RTC_DS3231 rtc;
Adafruit_TSL2561_Unified tsl = Adafruit_TSL2561_Unified(TSL2561_ADDR_FLOAT, 12345);

int valorSeco = 800;
int valorMojado = 350;

void setup() {
  Serial.begin(9600);
  dht.begin();

  if (!rtc.begin()) {
    Serial.println("No se detecto RTC");
    while (1);
  }

  if (!SD.begin(SD_CS)) {
    Serial.println("No se detecto microSD");
    while (1);
  }

  if (!tsl.begin()) {
    Serial.println("No se detecto TSL2561");
    while (1);
  }

  tsl.enableAutoRange(true);
  tsl.setIntegrationTime(TSL2561_INTEGRATIONTIME_402MS);

  Serial.println("Sistema iniciado");
}

void loop() {
  DateTime now = rtc.now();

  char filename[20];
  sprintf(filename, "%04d-%02d-%02d.csv", now.year(), now.month(), now.day());

  bool nuevoArchivo = !SD.exists(filename);

  File archivo = SD.open(filename, FILE_WRITE);

  if (archivo) {
    if (nuevoArchivo) {
      archivo.println("Fecha,Hora,Temperatura_C,Humedad_Ambiental_%,Humedad_Suelo_%,Luz_lux");
    }

    float temperatura = dht.readTemperature();
    float humedadAmbiental = dht.readHumidity();

    int lecturaSuelo = analogRead(SOIL_PIN);
    int humedadSuelo = map(lecturaSuelo, valorSeco, valorMojado, 0, 100);
    humedadSuelo = constrain(humedadSuelo, 0, 100);

    sensors_event_t event;
    tsl.getEvent(&event);

    float lux = event.light ? event.light : 0;

    archivo.print(now.year());
    archivo.print("-");
    archivo.print(now.month());
    archivo.print("-");
    archivo.print(now.day());
    archivo.print(",");

    archivo.print(now.hour());
    archivo.print(":");
    archivo.print(now.minute());
    archivo.print(":");
    archivo.print(now.second());
    archivo.print(",");

    archivo.print(temperatura);
    archivo.print(",");
    archivo.print(humedadAmbiental);
    archivo.print(",");
    archivo.print(humedadSuelo);
    archivo.print(",");
    archivo.println(lux);

    archivo.close();

    Serial.println("Datos guardados en SD");
  } else {
    Serial.println("Error al abrir archivo");
  }

  delay(17280000);
}
