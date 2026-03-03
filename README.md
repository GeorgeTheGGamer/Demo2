# TrackSense (Demo2)

Real-time lane, object, and rear foot safety monitoring for a mobile robot.

This project uses:
- **Laptop server** for AI inference, local CV display, control API, and live status streaming
- **Raspberry Pi client** for dual camera streaming + Arduino bridge
- **Mobile app** (React Native/Expo, in a sibling folder) for START/STOP and live telemetry

---

## Features

- Front camera lane detection + steering angle estimation
- Front object detection overlays (lane-aware box filtering)
- Rear camera lane + ankle/foot safety status
- Two local OpenCV windows on laptop (front + rear)
- App control via REST (`START` / `STOP`)
- Live status updates via WebSocket and `/status` endpoint
- UDP command forwarding to Pi and serial forwarding to Arduino

---

## Repository Structure

- `main.py` – project entry helper (if used)
- `tools/YOLOv8n/laptop_server.py` – central AI + API + display server
- `tools/YOLOv8n/pi_client.py` – Pi dual-stream sender + command listener
- `tools/YOLOv8n/objectDetector.py` – object detection helper
- `tools/YOLOv8n/poseDetector.py` – pose/ankle helper
- `tools/helper/steering_helper.py` – steering angle utilities
- `clrnet/` + `configs/` + `checkpoints/` – lane model code/config/weights

---

## Network & Ports

Default values in code:

- Laptop API host: `0.0.0.0`
- Laptop API port: `5050`
- Front frame UDP port: `8000`
- Rear frame UDP port: `8002`
- Pi command UDP port: `8001`
- Laptop IP used by app/Pi: `192.168.8.173`
- Pi IP targets in laptop server: `192.168.8.199`, `172.17.0.1`

If your network is different, update:
- `tools/YOLOv8n/laptop_server.py`
- `tools/YOLOv8n/pi_client.py`
- app endpoints in your Expo app

---

## Requirements

Use the provided requirements file for your OS:

- macOS: `requirements_mac.txt`
- Windows: `requirements_windows.txt`

Python 3.9+ recommended.

---

## Setup (Laptop)

1. Create and activate a virtual environment.
2. Install dependencies from your OS requirements file.
3. Ensure model files exist:
	- `checkpoints/tusimple_r18.pth`
	- `checkpoints/yolov8n_int8.tflite`
	- `checkpoints/yolov8n-pose_int8.tflite`
4. Run:

	`python3 tools/YOLOv8n/laptop_server.py`

Expected startup includes:
- front/rear UDP listeners
- Flask server at `http://<laptop-ip>:5050`
- OpenCV windows for front and rear feeds

---

## Setup (Raspberry Pi)

1. Connect cameras (`/dev/video0`, `/dev/video2` by default).
2. Connect Arduino serial (`/dev/ttyACM0` by default).
3. Ensure laptop IP in `pi_client.py` matches your laptop.
4. Run:

	`python3 tools/YOLOv8n/pi_client.py`

Pi waits for `START` command, then streams frames to laptop.

---

## Mobile App Flow

From the Expo app:
1. Press **START**
2. Laptop forwards START to Pi
3. Pi enables streaming
4. Laptop runs CV, shows overlays in both windows, and sends telemetry to app
5. Press **STOP** to stop robot streaming/control flow

---

## Runtime Behavior Notes

- Laptop keeps local windows visible (idle + running modes)
- Steering messages are sent as plain text: `ANGLE=<value>`
- START/STOP are plain commands over UDP to Pi
- App receives combined front/rear JSON state from laptop

---

## Troubleshooting

### 1) `Port 5000 is in use`
Current server port is `5050`. If you still see 5000 conflicts, another process/version is being run. Confirm you are launching the current `laptop_server.py`.

### 2) No feed in laptop windows
- Verify Pi is running and receives START
- Check `LAPTOP_IP` in `pi_client.py`
- Ensure UDP ports `8000`/`8002` are not blocked

### 3) App connects but no live updates
- Verify app uses `http://<laptop-ip>:5050`
- Verify WebSocket URL `ws://<laptop-ip>:5050/ws/status`
- Ensure laptop and phone are on the same network

### 4) Flask imports unresolved in editor
If IDE shows unresolved import for `flask`/`flask_sock`, verify interpreter selection points to the same virtual environment used to run the project.

### 5) Arduino not receiving commands
- Confirm serial port (`/dev/ttyACM0`) and baud (`115200`)
- Check user permissions for serial device
- Look for `[PI] -> ARDUINO:` logs in Pi terminal

---

## Safety

This code is for prototype/testing use. Validate all stop and steering behavior in a controlled environment before real operation.

