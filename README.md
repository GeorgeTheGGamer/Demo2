# TrackSense

TrackSense is an autonomous running companion system that pairs a mobile robot payload with an iOS/Android application. The system provides real-time computer vision analysis (lane detection, object avoidance, and rear foot tracking) alongside a mobile interface offering live GPS tracking, telemetry bridging, voice control, and automatic Strava uploads.

---

## System Architecture & Pipeline

The project operates across a distributed pipeline encompassing a companion computer (Raspberry Pi), a central inference server (Laptop), hardware controllers (Arduino), and a mobile application (React Native).

### 1. Hardware & ROS Node Layer (Raspberry Pi / Robot)
The robot is equipped with two camera sensors (front and rear) and a motor controller.
- **Sensor Nodes**: The ROS nodes (`front_camera_node.py` and `rear_camera_node.py`) capture raw video frames and stream them continuously via UDP over the local network to the AI server.
- **Actuation Nodes**: The ROS `bridge_node.py` and `arduino_node.py` listen for computed directional and throttle commands from the server and forward them sequentially to the Arduino.
- **Arduino Firmware**: The `arduino.ino` script receives structured serial commands from the Pi and physically actuates the robot's motor drivers.

### 2. Central AI Inference Server (Laptop)
The `laptop_server.py` script serves as the centralized computing core. It bridges the hardware layer and the mobile layer.
- **Stream Ingestion**: Binds to UDP sockets (ports `8000` and `8002`) to ingest continuous video frames from the robot.
- **Computer Vision Processing**: Runs multiple machine learning models concurrently on each frame to determine environmental safety and optimal trajectory.
- **Decision Engine**: Calculates the required steering angle (`ANGLE=<value>`) based on lane centers, checks for obstacle collision boundaries, and evaluates rear ankle proximity to determine if emergency stop conditions must be triggered.
- **API & Telemetry Bridging**: Exposes a Flask REST API (port `5050`) for accepting system `START` and `STOP` commands. It concurrently hosts a WebSocket server (`/ws/status`) that broadcasts the aggregated AI inference metrics to the mobile app at a high refresh rate.

### 3. Mobile Application (React Native / Expo)
The `SDP_TrackSense_App` repository contains the user-facing mobile client.
- **Telemetry Dashboard**: Subscribes to the central server's WebSocket to render live diagnostic data (e.g., Robot Status, Detected Objects, Stop Conditions).
- **Control Interface**: Allows users to initiate or terminate the robot's autonomous follow mode using either screen interactions or always-on voice commands utilizing speech recognition.
- **Location Tracking Module**: Subscribes to device-level high-accuracy GPS coordinates, applying spatial filters (requiring >5m movement, <20m accuracy radius) to eliminate stationary jitter and plot a geographic route Polyline on a MapView.
- **Strava Integration**: Facilitates secure OAuth2 authentication with Strava. Upon terminating a session, the app compiles the collected GPS coordinates into a standard GPX file schema and uploads the activity directly to the user's Strava account.

---

## Machine Learning Models

TrackSense utilizes three distinct models for its perception stack:

1. **CLRNet (Cross Layer Refinement Network) for Lane Detection**
   - **Architecture**: ResNet-18 backbone.
   - **Weights**: `tusimple_r18.pth`
   - **Purpose**: Detects the boundaries of the running path ahead of the robot. The server calculates the midpoint between the detected lane lines to generate proportional steering adjustments, ensuring the robot drives autonomously along the center of the track.

2. **YOLOv8 Nano for Object Detection**
   - **Architecture**: YOLOv8n (Quantized to INT8).
   - **Weights**: `yolov8n_int8.tflite`
   - **Purpose**: Scans the front camera feed for hazards (e.g., people, vehicles, animals). The pipeline cross-references the bounding boxes of detected objects with the physical lane boundaries calculated by CLRNet. If an object is inside the lane path, the robot triggers an avoidance or stop protocol.

3. **YOLOv8 Nano Pose for Foot Tracking**
   - **Architecture**: YOLOv8n-Pose (Quantized to INT8).
   - **Weights**: `yolov8n-pose_int8.tflite`
   - **Purpose**: Analyzes the rear camera feed to detect and track keypoints associated with human ankles and feet. This serves as a safety tether; if no feet are detected behind the robot, it assumes the runner is absent or has fallen, and halts immediately.

---

## Software Features Overview

- **Lane-Aware Filtering**: Obstacle detection is restricted exclusively to the traversable path, preventing false positive stops from objects off the track.
- **Multi-Modal App Control**: Start/Stop the tracking system via physical UI taps, prolonged "Hold-to-Stop" gestures, or hands-free voice commands ("TrackGo", "TrackStop") with robust debouncing locks.
- **Live Local Metric Calculations**: The mobile app calculates running distance, duration, and average pace (min/km) locally using the Haversine formula on accumulated GPS pings.
- **Private GPX Automations**: GPS routes are serialized into XML logic, and the Strava V3 Upload API payload enforces an `only_me` parameter by default.
- **Secure Key Storage**: Strava Access and Refresh OAuth tokens are encrypted natively using `expo-secure-store`.

---

## Setup Instructions

### 1. Laptop Server Configuration
1. Install Python 3.9+ requirements: `pip install -r requirements_mac.txt` (or Windows equivalent).
2. Ensure the pre-trained weights (`tusimple_r18.pth`, `yolov8n_int8.tflite`, `yolov8n-pose_int8.tflite`) are placed in the `/checkpoints` directory.
3. Start the server:
   ```bash
   python3 Demo3/laptop_server.py
   ```

### 2. Mobile App Deployment
1. Navigate into the application root: `cd SDP_TrackSense_App`.
2. Provide your Strava API keys by creating a `.env` file:
   ```env
   EXPO_PUBLIC_STRAVA_CLIENT_ID="your_client_id"
   EXPO_PUBLIC_STRAVA_CLIENT_SECRET="your_client_secret"
   ```
3. Boot the local bundler for development:
   ```bash
   npx expo start --dev-client
   ```

### 3. Robot/Hardware Deployment
1. Deploy the ROS network comprising `bridge_node.py` and the respective camera streaming nodes onto the Raspberry Pi.
2. Flash `arduino/arduino.ino` via the Arduino IDE to the connected stepper/motor control board.
3. Verify that the targeting `LAPTOP_IP` string in the ROS/Pi network properly maps to the static IPv4 address of the Central AI Inference Server.
