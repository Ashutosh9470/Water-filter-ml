
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>

// =====================================================
// 1. CONFIGURATION
// =====================================================

// WiFi credentials
const char* WIFI_SSID = "Ashutosh";
const char* WIFI_PASSWORD = "ashutosh01";

// FastAPI server
const char* SERVER_URL =
  "http://10.219.120.27:8000/sensor-data";

// Use a fresh ID to avoid mixing old simulated readings
const char* DEVICE_ID = "HORIZON_REAL_001";

// Sensor pins
#define FLOW_PIN 27
#define TEMP_PIN 4
#define TURBIDITY_PIN 34

// OLED pins
#define OLED_SDA 21
#define OLED_SCL 22

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

// YF-S201 provisional calibration
const float PULSES_PER_LITRE = 450.0;

// Data transmission interval
const unsigned long SEND_INTERVAL = 10000;

// =====================================================
// 2. SENSOR OBJECTS
// =====================================================

OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  OLED_RESET
);

// =====================================================
// 3. GLOBAL VARIABLES
// =====================================================

volatile unsigned long pulseCount = 0;

unsigned long intervalPulses = 0;
unsigned long lastSendTime = 0;

float temperature = NAN;
int turbidityRaw = 0;
float flowRate = 0;

float remainingLife = 0;
float remainingLitres = 0;
float cumulativeLitres = 0;
float remainingDays = -1;

bool serverConnected = false;
bool oledReady = false;

// =====================================================
// 4. FLOW SENSOR INTERRUPT
// =====================================================

void IRAM_ATTR countFlowPulse() {
  pulseCount++;
}

// =====================================================
// 5. WIFI CONNECTION
// =====================================================

void connectWiFi() {

  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("Connecting to WiFi...");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startTime = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - startTime < 20000) {

    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println("\nWiFi connected!");

    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());

  } else {

    Serial.println("\nWiFi connection failed!");
  }
}

// =====================================================
// 6. OLED DISPLAY
// =====================================================

void showMessage(const char* message) {

  if (!oledReady) return;

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.println("HORIZON SYSTEM");
  display.println("----------------");
  display.println(message);

  display.display();
}

void updateOLED() {

  if (!oledReady) return;

  display.clearDisplay();

  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);

  display.print("HORIZON | ");
  display.println(serverConnected ? "ONLINE" : "OFFLINE");

  display.println("----------------");

  display.print("Filter Life: ");
  display.print(remainingLife, 1);
  display.println("%");

  display.print("Remaining: ");
  display.print(remainingLitres, 1);
  display.println(" L");

  display.print("Days Left: ");

  if (remainingDays >= 0) {
    display.print(remainingDays, 0);
  } else {
    display.print("--");
  }

  display.println();

  display.print("Flow: ");
  display.print(flowRate, 2);
  display.println(" L/min");

  display.print("Temp: ");

  if (isnan(temperature)) {
    display.println("Sensor Error");
  } else {
    display.print(temperature, 1);
    display.println(" C");
  }

  display.print("Turbidity ADC: ");
  display.println(turbidityRaw);

  display.display();
}

// =====================================================
// 7. READ SENSORS
// =====================================================

void readSensors(float intervalSeconds) {

  // DS18B20 temperature
  tempSensor.requestTemperatures();

  float temp = tempSensor.getTempCByIndex(0);

  if (temp != DEVICE_DISCONNECTED_C && temp > -55 && temp < 125) {

    temperature = temp;

  } else {

    Serial.println("DS18B20 reading failed!");
  }

  // RM013404 analog output
  turbidityRaw = analogRead(TURBIDITY_PIN);

  // Capture and reset flow pulses atomically
  noInterrupts();
  intervalPulses = pulseCount;
  pulseCount = 0;
  interrupts();

  // Convert pulses into litres
  float litresProcessed =
    intervalPulses / PULSES_PER_LITRE;

  // Calculate flow rate
  flowRate =
    (litresProcessed / intervalSeconds) * 60.0;

  Serial.println("\n===== SENSOR READINGS =====");

  Serial.print("Flow pulses: ");
  Serial.println(intervalPulses);

  Serial.print("Flow rate: ");
  Serial.print(flowRate);
  Serial.println(" L/min");

  Serial.print("Temperature: ");
  Serial.println(temperature);

  Serial.print("Turbidity ADC: ");
  Serial.println(turbidityRaw);
}

// =====================================================
// 8. SEND DATA TO FASTAPI
// =====================================================

void sendSensorData(float intervalSeconds) {

  connectWiFi();

  if (WiFi.status() != WL_CONNECTED) {

    serverConnected = false;
    updateOLED();
    return;
  }

  WiFiClient client;
  HTTPClient http;

  http.begin(client, SERVER_URL);
  http.setTimeout(5000);

  http.addHeader("Content-Type", "application/json");

  // Build JSON request
  JsonDocument requestDoc;

  requestDoc["device_id"] = DEVICE_ID;
  requestDoc["pulse_count"] = intervalPulses;
  requestDoc["interval_seconds"] = intervalSeconds;

  // Send temperature only when valid
  if (!isnan(temperature)) {
    requestDoc["temperature"] = temperature;
  } else {
    http.end();
    Serial.println("Skipping request: invalid temperature.");
    serverConnected = false;
    updateOLED();
    return;
  }

  // TEMPORARY: raw ADC value in the existing field.
  // This is NOT calibrated NTU.
  requestDoc["outlet_turbidity"] = turbidityRaw;

  String payload;
  serializeJson(requestDoc, payload);

  Serial.println("\n===== SENDING TO SERVER =====");
  Serial.println(payload);

  int httpCode = http.POST(payload);

  if (httpCode > 0) {

    String response = http.getString();

    Serial.print("HTTP Response: ");
    Serial.println(httpCode);

    Serial.println("Server response:");
    Serial.println(response);

    if (httpCode == 200) {

      JsonDocument responseDoc;

      DeserializationError error =
        deserializeJson(responseDoc, response);

      if (!error) {

        serverConnected =
          (responseDoc["status"] == "success");

        JsonObject filter = responseDoc["filter"];

        if (!filter.isNull()) {

          remainingLife =
            filter["remaining_life_pct"] | remainingLife;

          remainingLitres =
            filter["remaining_litres"] | remainingLitres;

          cumulativeLitres =
            filter["cumulative_litres"] | cumulativeLitres;

          // Days will remain unavailable until
          // the backend sends a valid estimate.
          if (!filter["remaining_days"].isNull()) {

            remainingDays =
              filter["remaining_days"].as<float>();

          } else {

            remainingDays = -1;
          }
        }

      } else {

        Serial.println("JSON parsing failed!");
        serverConnected = false;
      }

    } else {

      serverConnected = false;
    }

  } else {

    Serial.print("HTTP request failed: ");
    Serial.println(http.errorToString(httpCode));

    serverConnected = false;
  }

  http.end();

  updateOLED();
}

// =====================================================
// 9. SETUP
// =====================================================

void setup() {

  Serial.begin(115200);
  delay(1000);

  Serial.println("\n============================");
  Serial.println("      HORIZON SYSTEM");
  Serial.println("============================");

  // Flow sensor
  pinMode(FLOW_PIN, INPUT_PULLUP);

  attachInterrupt(
    digitalPinToInterrupt(FLOW_PIN),
    countFlowPulse,
    FALLING
  );

  // Turbidity ADC
  analogReadResolution(12);

  analogSetPinAttenuation(
    TURBIDITY_PIN,
    ADC_11db
  );

  // Temperature sensor
  tempSensor.begin();

  // OLED initialization
  Wire.begin(OLED_SDA, OLED_SCL);

  oledReady = display.begin(
    SSD1306_SWITCHCAPVCC,
    OLED_ADDRESS
  );

  if (oledReady) {

    showMessage("Initializing...");

  } else {

    Serial.println("OLED initialization failed!");
  }

  // Connect WiFi
  connectWiFi();

  lastSendTime = millis();
}

// =====================================================
// 10. MAIN LOOP
// =====================================================

void loop() {

  unsigned long currentTime = millis();

  if (currentTime - lastSendTime >= SEND_INTERVAL) {

    float elapsedSeconds =
      (currentTime - lastSendTime) / 1000.0;

    lastSendTime = currentTime;

    // Read all sensors
    readSensors(elapsedSeconds);

    // Send the same interval's readings to server
    sendSensorData(elapsedSeconds);
  }
}
