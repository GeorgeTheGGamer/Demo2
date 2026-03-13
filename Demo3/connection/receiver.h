#include <WiFi.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

const char* SERVER_IP = "192.168.1.100";
const uint16_t SERVER_PORT = 9001;

WiFiClient client;
String rxLine;

// =========================
// Optional output pins
// =========================
const int LEFT_PIN  = 25;
const int RIGHT_PIN = 26;

// =========================
// State enums
// =========================
enum BranchType {
  BRANCH_NONE,
  BRANCH_FEET,
  BRANCH_STEER
};

enum FeetState {
  FEET_UNKNOWN,
  FEET_INSIDE,
  FEET_LEFT_OUT,
  FEET_RIGHT_OUT,
  FEET_BOTH_OUT,
  FEET_NO_FEET
};

enum SteerState {
  STEER_UNKNOWN,
  STEER_NONE,
  STEER_LEFT,
  STEER_RIGHT
};

// -------------------------
// Helpers: normalize string
// -------------------------
String normalizeText(String s) {
  s.trim();
  s.toUpperCase();

  s.replace("_", " ");
  s.replace("-", " ");
  while (s.indexOf("  ") >= 0) {
    s.replace("  ", " ");
  }

  return s;
}

// -------------------------
// Parse rear_status -> FeetState
// -------------------------
FeetState parseFeetState(String rearStatusRaw) {
  String s = normalizeText(rearStatusRaw);

  if (s == "INSIDE") {
    return FEET_INSIDE;
  }
  if (s == "LEFT OUT") {
    return FEET_LEFT_OUT;
  }
  if (s == "RIGHT OUT" ) {
    return FEET_RIGHT_OUT;
  }
  if (s == "BOTH OUT") {
    return FEET_BOTH_OUT;
  }
  if (s == "NO FEET DETECTED") {
    return FEET_NO_FEET;
  }

  return FEET_UNKNOWN;
}

// -------------------------
// Parse lane_status -> SteerState
// -------------------------
SteerState parseSteerState(String laneStatusRaw) {
  String s = normalizeText(laneStatusRaw);

  if (s == "LEFT") {
    return STEER_LEFT;
  }
  if (s == "RIGHT") {
    return STEER_RIGHT;
  }
  if (s == "STRAIGHT") {
    return STEER_NONE;
  }

  return STEER_UNKNOWN;
}

// -------------------------
// Output actions
// TODO: replace with actual outputs (e.g. buzzer, vibration motor, etc.)
// -------------------------
void clearOutputs() {
  digitalWrite(LEFT_PIN, LOW);
  digitalWrite(RIGHT_PIN, LOW);
}

void runFeetBranch(FeetState feetState) {
  Serial.println("[ESP32] >>> FEET branch");

  switch (feetState) {
    case FEET_LEFT_OUT:
      Serial.println("[ESP32] FEET = LEFT_OUT");
      digitalWrite(LEFT_PIN, HIGH);
      digitalWrite(RIGHT_PIN, LOW);
      break;

    case FEET_RIGHT_OUT:
      Serial.println("[ESP32] FEET = RIGHT_OUT");
      digitalWrite(LEFT_PIN, LOW);
      digitalWrite(RIGHT_PIN, HIGH);
      break;

    case FEET_BOTH_OUT:
      Serial.println("[ESP32] FEET = BOTH_OUT");
      digitalWrite(LEFT_PIN, HIGH);
      digitalWrite(RIGHT_PIN, HIGH);
      break;

    case FEET_NO_FEET:
      Serial.println("[ESP32] FEET = NO_FEET");
      digitalWrite(LEFT_PIN, HIGH);
      digitalWrite(RIGHT_PIN, HIGH);
      break;

    case FEET_INSIDE:
      Serial.println("[ESP32] FEET = INSIDE");
      clearOutputs();
      break;

    default:
      Serial.println("[ESP32] FEET = UNKNOWN");
      clearOutputs();
      break;
  }
}

void runSteerBranch(SteerState steerState) {
  Serial.println("[ESP32] >>> STEER branch");

  switch (steerState) {
    case STEER_LEFT:
      Serial.println("[ESP32] STEER = LEFT");
      digitalWrite(LEFT_PIN, HIGH);
      digitalWrite(RIGHT_PIN, LOW);
      break;

    case STEER_RIGHT:
      Serial.println("[ESP32] STEER = RIGHT");
      digitalWrite(LEFT_PIN, LOW);
      digitalWrite(RIGHT_PIN, HIGH);
      break;

    case STEER_NONE:
      Serial.println("[ESP32] STEER = NONE/STRAIGHT");
      clearOutputs();
      break;

    default:
      Serial.println("[ESP32] STEER = UNKNOWN");
      clearOutputs();
      break;
  }
}

// -------------------------
// Decide which big branch to enter
// -------------------------
void handlePayload(const String& jsonLine) {
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, jsonLine);

  if (err) {
    Serial.print("[ESP32] JSON parse failed: ");
    Serial.println(err.c_str());
    return;
  }

  String rearStatus = doc["rear_status"] | "";
  String laneStatus = doc["lane_status"] | "";

  Serial.print("[ESP32] rear_status = ");
  Serial.println(rearStatus);
  Serial.print("[ESP32] lane_status = ");
  Serial.println(laneStatus);

  FeetState feetState = parseFeetState(rearStatus);
  SteerState steerState = parseSteerState(laneStatus);

  // priority：
  // 1. feet error
  // 2. feet inside then steer error
  // 3. safe
  if (
      feetState == FEET_LEFT_OUT ||
      feetState == FEET_RIGHT_OUT ||
      feetState == FEET_BOTH_OUT ||
      feetState == FEET_NO_FEET
     ) {
    runFeetBranch(feetState);
    return;
  }

  if (feetState == FEET_INSIDE) {
    if (steerState == STEER_LEFT || steerState == STEER_RIGHT) {
      runSteerBranch(steerState);
      return;
    } else {
      Serial.println("[ESP32] FEET inside + no steer action");
      clearOutputs();
      return;
    }
  }

  // unknown state or error
  Serial.println("[ESP32] Unknown payload state, outputs cleared");
  clearOutputs();
}

// Connect to WiFi; blocks until connected. Retries every ~10s if failed.
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("[ESP32] Connecting WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int retry = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    retry++;
    if (retry > 20) { // ~10s
      Serial.println("\n[ESP32] WiFi retry...");
      WiFi.disconnect(true);
      delay(500);
      WiFi.begin(WIFI_SSID, WIFI_PASS);
      retry = 0;
    }
  }
  Serial.printf("\n[ESP32] WiFi connected, IP=%s\n", WiFi.localIP().toString().c_str());
}

// Connect to laptop TCP server; returns true if connected.
bool connectServer() {
  if (client.connected()) return true;

  Serial.printf("[ESP32] Connecting server %s:%u ...\n", SERVER_IP, SERVER_PORT);
  if (!client.connect(SERVER_IP, SERVER_PORT)) {
    Serial.println("[ESP32] Server connect failed");
    return false;
  }

  client.setNoDelay(true);
  client.setTimeout(20);
  Serial.println("[ESP32] Server connected");
  return true;
}

// Read newline-delimited JSON messages from laptop
void readServerLines() {
  while (client.connected() && client.available()) {
    char c = (char)client.read();

    if (c == '\n') {
      rxLine.trim();
      if (rxLine.length() > 0) {
        Serial.print("[ESP32] <- ");
        Serial.println(rxLine);
        handlePayload(rxLine);
      }
      rxLine = "";
    } else {
      rxLine += c;

      // Guard against malformed long frames causing String growth
      if (rxLine.length() > 1024) {
        Serial.println("[ESP32] Oversized frame dropped");
        rxLine = "";
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(LEFT_PIN, OUTPUT);
  pinMode(RIGHT_PIN, OUTPUT);
  clearOutputs();

  ensureWiFi();
}

void loop() {
  ensureWiFi();

  if (!connectServer()) {
    delay(1000);
    return;
  }

  readServerLines();

  if (!client.connected()) {
    Serial.println("[ESP32] Server disconnected");
    client.stop();
    clearOutputs();
    delay(500);
  }

  delay(10);
}