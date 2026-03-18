#include <Arduino.h>
#include <Wire.h>
#include <TFLI2C.h>
#include <Grove_I2C_Motor_Driver.h>
#include <Servo.h>

#define I2C_ADDRESS      0x0f
#define ADC_MIC          A0
#define FRONT_SERVO_PIN  5
#define REAR_SERVO_PIN   8
#define LIDAR_ADDR       0x10

// TF-Luna returns this value when it has no valid reading
#define LIDAR_ERROR_VAL  9000

TFLI2C sensor;

Servo servoMotor;
Servo servoMotorLidar;

const int   STEADY_DISTANCE_CM = 200;
const int   STOP_DISTANCE_CM   = 250;
const int   MIN_SPEED          = 180;
const unsigned int CHECK_MS    = 100;

unsigned long lastCheckMs = 0;
unsigned long now         = 0;

int motorSpeed          = 0;
int servoMotorAngle     = 90;
int servoMotorLidarAngle = 90;

// --- State flag: replaces the blocking startBot() loop ---
bool isRunning = false;

// -------------------------------------------------------
// Speed calculation based on LiDAR distance
// -------------------------------------------------------
int calculateSpeed(int currentSpeed, int distance) {
    const float sensitivity = 0.75f;
    const float minGain     = 0.2f;
    const float maxGain     = 3.0f;

    float robotDist = (float)(distance - STEADY_DISTANCE_CM) / STEADY_DISTANCE_CM;
    float gain      = 1.0f - sensitivity * robotDist;
    gain            = constrain(gain, minGain, maxGain);

    int newSpeed = (int)(currentSpeed * gain);
    return constrain(newSpeed, MIN_SPEED, 255);
}

// -------------------------------------------------------
// Servo helpers — only write if angle changed
// -------------------------------------------------------
void steerBot(int angle) {
    angle = constrain(angle, 60, 120);   // Bug 4: guard against out-of-range serial values
    if (angle == servoMotorAngle) return;
    servoMotor.write(angle);
    servoMotorAngle = angle;
}

void steerLidar(int angle) {
    angle = constrain(angle, 45, 135);   // Bug 4: guard against out-of-range serial values
    if (angle == servoMotorLidarAngle) return;
    servoMotorLidar.write(angle);
    servoMotorLidarAngle = angle;
}

// -------------------------------------------------------
// Start / stop helpers — no longer blocking
// -------------------------------------------------------
void startRobot() {
    steerBot(90);
    steerLidar(90);
    Motor.speed(MOTOR1, 150);
    motorSpeed   = 150;
    lastCheckMs  = millis();
    isRunning    = true;
    Serial.println("Robot started");
}

void stopRobot() {
    Motor.speed(MOTOR1, 0);
    motorSpeed = 0;
    steerBot(90);
    steerLidar(90);
    isRunning = false;
    Serial.println("Robot stopped");
}

// -------------------------------------------------------
// Command dispatcher
// -------------------------------------------------------
void receiveCommand(const String& command) {
    if (command == "START") {
        startRobot();
        return;
    }

    if (command == "STOP") {
        stopRobot();  // Bug 1: no longer calls startBot() / blocks here
        return;
    }

    if (command.startsWith("FRONT_ANGLE=")) {
        int front_angle = 90;
        int rear_angle  = 90;

        int parsed_count = sscanf(command.c_str(),
                                  "FRONT_ANGLE=%d,REAR_ANGLE=%d",
                                  &front_angle, &rear_angle);

        if (parsed_count >= 1) {         // Bug 5: handle partial parse too
            steerBot(front_angle);
        }
        if (parsed_count == 2) {
            steerLidar(rear_angle);
        }
    }
}

// -------------------------------------------------------
// Setup
// -------------------------------------------------------
void setup() {
    Serial.begin(115200);
    Serial.setTimeout(50);   // Bug 6: short timeout so loop() isn't blocked for 1s

    Wire.begin();
    Motor.begin(I2C_ADDRESS);
    Motor.speed(MOTOR1, 0);

    servoMotor.attach(FRONT_SERVO_PIN);
    servoMotorLidar.attach(REAR_SERVO_PIN);

    steerBot(90);
    steerLidar(90);

    // Bug 2: removed blocking startBot() — wait for START command via loop()
    Serial.println("Bridge ready. Waiting for START command.");
}

// -------------------------------------------------------
// Main loop — non-blocking
// -------------------------------------------------------
void loop() {
    // Read and dispatch any incoming serial command
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        if (command.length() > 0) {
            receiveCommand(command);
        }
    }

    // LiDAR distance check — only when robot is running
    if (isRunning) {
        now = millis();
        if (now - lastCheckMs >= CHECK_MS) {
            int dist = 0;
            if (sensor.getData(dist, LIDAR_ADDR)) {
                if (dist == LIDAR_ERROR_VAL || dist <= 0) {
                    // Bug 7: treat known error values as no-data — don't adjust speed
                    Serial.println("LiDAR: no valid reading");
                } else if (dist >= STOP_DISTANCE_CM) {
                    motorSpeed = max(motorSpeed - 5, MIN_SPEED);
                    Motor.speed(MOTOR1, motorSpeed);
                } else {
                    motorSpeed = calculateSpeed(motorSpeed, dist);
                    Motor.speed(MOTOR1, motorSpeed);
                }
            } else {
                Serial.println("LiDAR: sensor read failed");
            }
            lastCheckMs = now;
        }
    }
}