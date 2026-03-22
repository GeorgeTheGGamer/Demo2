#include <Arduino.h>
#include <Grove_I2C_Motor_Driver.h>
#include <Servo.h>
#include <TFLI2C.h>
#include <Wire.h>

#define I2C_ADDRESS 0x0f
#define ADC_MIC A0
#define FRONT_SERVO_PIN 5
#define REAR_SERVO_PIN 8

TFLI2C sensor;

Servo servoMotor;
Servo servoMotorLidar;

const double EPSILON{0.001};

const int STEADY_DISTANCE_CM{200};
const unsigned int CHECK_MS{200};
const int STOP_DISTANCE_CM{250};
const int START_VOLUME{200};
const int MIN_SPEED{180};
#define LIDAR_ERROR_VAL 9000

unsigned long lastCheckMs{};
unsigned long now{};

int motorSpeed{};
int servoMotorAngle{};
int servoMotorLidarAngle{};
bool isRunning = false;
String commandBuf = "";

int calculateSpeed(int currentSpeed, int distance) {
  const float sensitivity = 0.75f;
  const float minGain = 0.2f;
  const float maxGain = 3.0f;

  float robotDist = (float)(distance - STEADY_DISTANCE_CM) / STEADY_DISTANCE_CM;
  float gain = 1.0f - sensitivity * robotDist;
  gain = constrain(gain, minGain, maxGain);

  int newSpeed = (int)(currentSpeed * gain);
  return constrain(newSpeed, MIN_SPEED, 255);
}

void steerBot(int angle) {
  angle = constrain(angle, 60, 120); // Safety bounds
  if (angle == servoMotorAngle)
    return;
  servoMotor.write(angle);
  servoMotorAngle = angle;
}

void steerLidar(int angle) {
  angle = constrain(angle, 45, 135); // Safety bounds
  if (angle == servoMotorLidarAngle)
    return;
  servoMotorLidar.write(angle);
  servoMotorLidarAngle = angle;
}

void startRobot() {
  isRunning = true;
  Serial.println("1"); // Robot starting via App
  steerBot(90);
  steerLidar(90);
  Motor.speed(MOTOR1, 150);
  motorSpeed = 150;
  lastCheckMs = millis();
}

void stopRobot() {
  isRunning = false;
  Serial.println("0"); // Robot stopping via App
  for (int i{}; i < 10; ++i) {
    Motor.stop(MOTOR1);
    Motor.speed(MOTOR1, 0);
    delay(5);
  }
  motorSpeed = 0;
  steerBot(90);
  steerLidar(90);
}

void receiveCommand(const String& command) {

  if (command.indexOf("STOP") != -1 || command.indexOf("START") != -1) {
    Serial.print("DBG received: [");
    Serial.print(command);
    Serial.println("]");
  }

  if (command.indexOf("START") != -1) {
    if (!isRunning) {
      startRobot();
    }
  } else if (command.indexOf("STOP") != -1) {
    if (isRunning) {
      stopRobot();
    }
  } else if (command.startsWith("FRONT_ANGLE=")) {
    int front_angle = 90;
    int rear_angle = 90;
    int parsed_count = sscanf(command.c_str(), "FRONT_ANGLE=%d,REAR_ANGLE=%d",
                              &front_angle, &rear_angle);
    if (parsed_count >= 1)
      steerBot(front_angle);
    if (parsed_count == 2)
      steerLidar(rear_angle);
  }
}

void setup() {
  Serial.begin(115200);
  commandBuf.reserve(64); // Prevent heap fragmentation
  Wire.begin();
  Wire.setWireTimeout(3000, true); // Prevent I2C from hanging the Arduino
  Motor.begin(I2C_ADDRESS);
  Motor.speed(MOTOR1, 0);

  servoMotor.attach(FRONT_SERVO_PIN);
  servoMotorLidar.attach(REAR_SERVO_PIN);

  steerBot(90);
  steerLidar(90);

  Serial.println("DBG: setup complete, entering loop");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      commandBuf.trim();
      if (commandBuf.length() > 0) {
        receiveCommand(commandBuf);
      }
      commandBuf = "";
    } else {
      // Don't overflow the buffer
      if (commandBuf.length() < 63) {
        commandBuf += c;
      }
    }
  }

  if (!isRunning)
    return;

  //  if (!isRunning) {
  //    // Standard hardware listen for mic
  //    long sum = 0;
  //    sum >>= 4; // Mock mic read
  //    if (sum > START_VOLUME) {
  //      startRobot();
  //    }
  //    return; // Don't run LiDAR checks while stopped
  //  }

  now = millis();
  if (now - lastCheckMs >= CHECK_MS) {
    int dist;
    if (sensor.getData(dist, 0x10)) {
      if ((dist >= STOP_DISTANCE_CM && dist != LIDAR_ERROR_VAL) || dist <= 0) {
        motorSpeed = max(motorSpeed - 5, MIN_SPEED);
        Motor.speed(MOTOR1, motorSpeed);
      } else {
        motorSpeed = calculateSpeed(motorSpeed, dist);
        Motor.speed(MOTOR1, motorSpeed);
      }
    } else {
      Serial.println("No output from tf luna");
    }
    lastCheckMs = now;
  }
}
