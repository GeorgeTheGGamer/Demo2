# TrackSense

Real-time lane, object, and rear foot safety monitoring for a mobile robot.

TrackSense is an autonomous running companion system that pairs a mobile robot with an iOS/Android App. The robot provides live CV analysis (lane detection, object avoidance, foot tracking) and the mobile app provides live GPS tracking, telemetry, voice control, and automatic Strava uploads.

---

## 🚀 Key Features

### 🤖 Robot Intelligence (AI/CV)
- **Front Camera**: Live lane detection and steering angle estimation to stay on paths.
- **Object Detection**: YOLOv8n object detection with lane-aware bounding boxes to warn of upcoming hazards.
- **Rear Camera**: Ankle/foot tracking to ensure the robot stays ahead of the runner safely.
- **ROS Integration**: Modular architecture using ROS nodes for sensors and motor controllers.

### 📱 Mobile App (React Native/Expo)
- **Live Telemetry & GPS Tracking**: Real-time map viewport locked to the user's location, with distance and pace calculations (includes anti-drift jitter filtering).
- **Voice Control**: "TrackGo" and "TrackStop" voice commands with robust debouncing to navigate menus completely hands-free.
- **Strava Integration**: Built-in Strava OAuth support securely saves tokens (`expo-secure-store`). Automatically generates GPX files of the run and uploads them directly to your Strava profile natively. 
- **Gestures**: "Hold-to-Stop" gesture anywhere on the screen for 3 seconds guarantees an easy stop without needing to look at small buttons.

---

## 📂 Repository Structure

- `Demo3/laptop_server.py` – Central AI inference server, Flask API, and WebSocket server.
- `SDP_TrackSense_App/` – The React Native/Expo mobile app source code.
- `ros/` – ROS Nodes (`arduino_node.py`, `bridge_node.py`, `front_camera_node.py`, `rear_camera_node.py`).
- `arduino/arduino.ino` – The serial bridge script for the motor controller.
- `tools/helper/` & `clrnet/` – Model weights and steering calculation utilities.

---

## ⚙️ Setup Instructions

### 1. Laptop Server (AI Core)
1. Install dependencies from `requirements_mac.txt` or `requirements_windows.txt`.
2. Ensure model files exist in `checkpoints/`.
3. Run the AI server:
   ```bash
   python3 Demo3/laptop_server.py
   ```
   *(This launches the Flask instance on port `5050` and listens for UDP frames on `8000/8002`)*

### 2. Mobile App
1. Navigate to the App directory: `cd SDP_TrackSense_App`
2. Configure Strava API credentials:
   Create a `.env` file containing:
   ```env
   EXPO_PUBLIC_STRAVA_CLIENT_ID="your_id"
   EXPO_PUBLIC_STRAVA_CLIENT_SECRET="your_secret"
   ```
3. Start the Expo bundler:
   ```bash
   npx expo start --dev-client
   ```
   *(To test on a physical iOS device natively, use `npx expo run:ios -d` via Xcode if you encounter `devicectl` bugs).*

### 3. Robot/Raspberry Pi
1. Run the ROS nodes or Pi streaming clients (`tools/YOLOv8n/pi_client.py`).
2. Flash the `arduino/arduino.ino` to the motor controller to handle incoming serial directions.

---

## 📡 Networking & Ports
- **Laptop API host**: `0.0.0.0:5050`
- **Front frame UDP port**: `8000`
- **Rear frame UDP port**: `8002`
- **REST & WebSockets**: App connects via `/status` and `/command`.

*Ensure the LAPTOP_IP is correctly matched across `constants.js` in the app and the `pi_client.py` on the robot.*
