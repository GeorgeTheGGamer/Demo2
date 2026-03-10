import os
import sys
import json
import cv2
import time
import math
import torch
import socket
import importlib
import threading
import numpy as np
from flask import Flask, request, jsonify
from flask_sock import Sock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- AI & AUTONOMY IMPORTS ---
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
import objectDetector as oD
import poseDetector as pD
from tools.helper.steering_helper import SteeringHelper

# --- CONFIGURATION ---
HOST_IP = "0.0.0.0"
API_PORT = 5050
FRONT_PORT = 8000
REAR_PORT = 8002
MAX_DGRAM = 65507

PI_IPS = ["192.168.118.199"]
PI_CMD_PORT = 8001

DEFAULT_CONFIG = 'configs/clrnet/clr_resnet18_tusimple.py'
DEFAULT_CHECKPOINT = 'checkpoints/tusimple_r18.pth'
DEFAULT_DEVICE = 'mps' # Change to 'cuda' or 'cpu' as needed
DEFAULT_FRONT_YOLO = 'checkpoints/yolov8n_int8.tflite'
DEFAULT_REAR_POSE = 'checkpoints/yolov8n-pose_int8.tflite'

# --- GLOBALS ---
latest_rear_frame = None
latest_front_frame = None
is_running = False
cv_ready = False
cv_ready_event = threading.Event()
pi_started = False
condition_since = {}   # { 'front:<cond>' | 'rear:<cond>' : first_seen_timestamp }
last_auto_stop_send_ts = 0.0

tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
app = Flask(__name__)
sock = Sock(app)

state_lock = threading.Lock()
ws_lock = threading.Lock()
ws_clients = set()
last_ws_push_ts = 0.0
WS_PUSH_HZ = 5.0
AUTO_STOP_COOLDOWN_SEC = 1.0

# --- STOP CONDITION HOLD TIMES (seconds) ---
# How long a condition must be continuously active before STOP is sent to Pi.
# Raise for more leniency, lower for faster response.
HOLD_OBJECT_IN_LANE_SEC    = 10.0   # Front: YOLO object inside lane
HOLD_CORNER_ANGLE_SEC      = 6.0   # Front: heading angle >= 70 degrees
HOLD_FRONT_NO_LANE_SEC     = 8.0   # Front: CLRNet detects 0 lanes
HOLD_FRONT_OUT_LANE_SEC    = 8.0   # Front: robot out of lane
HOLD_LEFT_FOOT_SEC         = 10.0   # Rear: left ankle outside lane
HOLD_RIGHT_FOOT_SEC        = 10.0   # Rear: right ankle outside lane
HOLD_BOTH_FEET_SEC         = 10.0   # Rear: both ankles outside lane
HOLD_NO_FEET_SEC           = 15.0   # Rear: no ankles detected at all
HOLD_REAR_NO_LANE_SEC      = 8.0   # Rear: CLRNet detects 0 lanes
HOLD_REAR_OUT_LANE_SEC     = 8.0   # Rear: robot out of lane

latest_state = {
    'running': False,
    'auto_stop_reason': [],
    'front': {
        'robot_status': 'NORMAL',
        'ANGLE': 'ANGLE=0.00',
        'object_detection': {'warning': [], 'danger': []},
        'stop_conditions': [],
    },
    'rear': {
        'status': 'No feet detected',
        'object_detection': {'warning': [], 'danger': []},
        'stop_conditions': [],
    },
}

# --- HELPER FUNCTIONS ---
def resolve_path(path):
    if os.path.isabs(path): return path
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate): return cwd_candidate
    root_candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_candidate): return root_candidate
    return cwd_candidate

def choose_device(device_flag):
    if device_flag == 'cuda' or torch.cuda.is_available(): return torch.device('cuda')
    if device_flag == 'mps' or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()): return torch.device('mps')
    return torch.device('cpu')

def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    cleaned = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

def preprocess_frame(frame, cfg, device):
    frame = cv2.resize(frame, (cfg.ori_img_w, cfg.ori_img_h), interpolation=cv2.INTER_LINEAR)
    cropped = frame[cfg.cut_height:, :, :]
    resized = cv2.resize(cropped, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return frame, tensor

def draw_lanes(frame, lanes, cfg, line_width=4):
    lanes_xy = []
    h, w = frame.shape[:2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, lane in enumerate(lanes):
        pts = lane.to_array(cfg)
        xy = [(int(round(p[0])), int(round(p[1]))) for p in pts if 0 <= p[0] < w and 0 <= p[1] < h]
        if len(xy) >= 2:
            lanes_xy.append(xy)
            color = colors[i % len(colors)]
            for j in range(1, len(xy)):
                cv2.line(frame, xy[j - 1], xy[j], color, thickness=line_width)
    return lanes_xy


def interpolate_x_at_y(polyline, y):
    for i in range(1, len(polyline)):
        x1, y1 = polyline[i - 1]
        x2, y2 = polyline[i]
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        if y_min <= y <= y_max and y1 != y2:
            t = (y - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)
    return None


def is_point_in_lane(pt, lanes_xy):
    if pt is None or len(lanes_xy) < 2:
        return False
    cx, cy = pt
    xs_at_y = []
    for lane_xy in lanes_xy:
        x = interpolate_x_at_y(lane_xy, cy)
        if x is not None:
            xs_at_y.append(x)
    if len(xs_at_y) < 2:
        return False
    xs_at_y.sort()
    for i in range(0, len(xs_at_y) - 1):
        if xs_at_y[i] <= cx <= xs_at_y[i + 1]:
            return True
    return False


def normalize_point(pt):
    if pt is None:
        return None
    return float(pt[0]), float(pt[1])


def feet_status(left_ankle, right_ankle, lanes_xy):
    left = normalize_point(left_ankle)
    right = normalize_point(right_ankle)
    left_in = False if left is None else is_point_in_lane(left, lanes_xy)
    right_in = False if right is None else is_point_in_lane(right, lanes_xy)
    if left is None and right is None:
        return 'No feet detected', left_in, right_in
    if left_in and right_in:
        return 'Safe', left_in, right_in
    if (not left_in) and right_in:
        return 'Left out', left_in, right_in
    if left_in and (not right_in):
        return 'Right out', left_in, right_in
    return 'Both out', left_in, right_in


def build_front_detection(objects, lanes_xy, frame_shape, names=None, close_ratio=0.7):
    warning = []
    danger = []
    h, w = frame_shape[:2]
    frame_area = max(1.0, float(h * w))

    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        in_lane = (
            is_point_in_lane((x1, y2), lanes_xy)
            or is_point_in_lane(((x1 + x2) / 2.0, y2), lanes_xy)
            or is_point_in_lane((x2, y2), lanes_xy)
        )
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        is_close = ((box_w * box_h) / frame_area) >= close_ratio

        cls_id = obj['cls']
        raw_name = names[cls_id] if (names is not None and cls_id in names) else str(cls_id)
        name = 'person' if raw_name == 'person' else 'object'
        side = 'left' if ((x1 + x2) / 2.0) < (w / 2.0) else 'right'
        label = f'{name}({side})'

        if in_lane:
            danger.append(label)
        elif is_close:
            warning.append(label)

    return {'warning': warning, 'danger': danger}


def build_rear_detection(status):
    warning = []
    danger = []
    if status == 'No feet detected':
        warning.append('person(unknown_ankles)')
    elif status == 'Left out':
        danger.append('left_foot(out_of_lane)')
    elif status == 'Right out':
        danger.append('right_foot(out_of_lane)')
    elif status == 'Both out':
        danger.append('left_foot(out_of_lane)')
        danger.append('right_foot(out_of_lane)')
    return {'warning': warning, 'danger': danger}


def build_status_payload():
    return {
        'running': latest_state.get('running', False),
        'cv_ready': cv_ready,
        'auto_stop_reason': latest_state.get('auto_stop_reason', []),
        'front': {
            'robot_status': latest_state.get('front', {}).get('robot_status', 'NORMAL'),
            'ANGLE': latest_state.get('front', {}).get('ANGLE', 'ANGLE=0.00'),
            'object_detection': latest_state.get('front', {}).get('object_detection', {'warning': [], 'danger': []}),
            'stop_conditions': latest_state.get('front', {}).get('stop_conditions', []),
        },
        'rear': {
            'status': latest_state.get('rear', {}).get('status', 'No feet detected'),
            'object_detection': latest_state.get('rear', {}).get('object_detection', {'warning': [], 'danger': []}),
            'stop_conditions': latest_state.get('rear', {}).get('stop_conditions', []),
        },
    }


def broadcast_status(force=False):
    global last_ws_push_ts
    with state_lock:
        payload = build_status_payload()

    if (not force) and (not payload.get('running', False)):
        return

    now = time.time()
    if (not force) and ((now - last_ws_push_ts) < (1.0 / WS_PUSH_HZ)):
        return
    last_ws_push_ts = now

    serialized = json.dumps(payload)
    dead = []
    with ws_lock:
        for ws in ws_clients:
            try:
                ws.send(serialized)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.discard(ws)


def forward_command_to_pi(command):
    """Forward START/STOP command to all configured Pi IPs."""
    payload = f"{command}\n".encode('utf-8')
    for pi_ip in PI_IPS:
        print(f"[PI CMD] {command} -> {pi_ip}:{PI_CMD_PORT}")
        tx_socket.sendto(payload, (pi_ip, PI_CMD_PORT))

# --- NETWORKING THREADS ---
@app.route('/command', methods=['POST'])
def receive_command():
    global is_running
    command = (request.json or {}).get('action', '').strip().upper()

    if command == 'START':
        global pi_started, condition_since, last_auto_stop_send_ts
        is_running = True
        pi_started = False          # Reset so Pi gets START on first angle of this new run
        condition_since = {}        # Clear all hold timers from previous run
        last_auto_stop_send_ts = 0.0  # Reset cooldown so first stop of new run fires immediately
        print('📱 APP SAYS START: CV started locally — Pi will be notified on first angle')
        with state_lock:
            latest_state['running'] = True
        broadcast_status(force=True)
        return jsonify({'status': 'success', 'running': True}), 200

    if command == 'STOP':
        is_running = False
        print('📱 APP SAYS STOP: Forwarding to Pi...')
        forward_command_to_pi('STOP')
        with state_lock:
            latest_state['running'] = False
        broadcast_status(force=True)
        return jsonify({'status': 'success', 'running': False}), 200

    return jsonify({'status': 'error'}), 400


@app.route('/status', methods=['GET'])
def get_status():
    with state_lock:
        return jsonify(build_status_payload()), 200


@sock.route('/ws/status')
def ws_status(ws):
    with ws_lock:
        ws_clients.add(ws)
    try:
        with state_lock:
            ws.send(json.dumps(build_status_payload()))
        while True:
            msg = ws.receive()
            if msg is None:
                break
    finally:
        with ws_lock:
            ws_clients.discard(ws)

def run_tcp_server():
    app.run(host='0.0.0.0', port=API_PORT, debug=False, use_reloader=False)

def receive_rear_video():
    global latest_rear_frame
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST_IP, REAR_PORT))
    print(f"[SERVER] 🟢 Listening for REAR camera on port {REAR_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(MAX_DGRAM)
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None: latest_rear_frame = frame
        except Exception as e:
            pass

def receive_front_video():
    global latest_front_frame
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST_IP, FRONT_PORT))
    print(f"[SERVER] 🔵 Listening for FRONT camera on port {FRONT_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(MAX_DGRAM)
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None: latest_front_frame = frame
        except Exception as e:
            pass

def send_angle_to_pi(angle_deg):
    global pi_started
    # On the very first angle of a run, send START to Pi first so it is ready to move.
    # Guard with is_running to avoid a race where STOP arrives mid-loop.
    if not pi_started:
        if not is_running:
            return
        print('[PI CMD] First angle computed — sending START to Pi now')
        forward_command_to_pi('START')
        pi_started = True
    # Sends plain-text steering to Pi (no JSON)
    body = f"ANGLE={angle_deg:.2f}"
    for pi_ip in PI_IPS:
        print(f"[PI STEER] {body} -> {pi_ip}:{PI_CMD_PORT}")
        tx_socket.sendto(body.encode('utf-8'), (pi_ip, PI_CMD_PORT))


def make_placeholder_frame(title, width=960, height=540):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, title, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(frame, 'Waiting for stream...', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    return frame


def is_object_close_to_lane(obj, lanes_xy, distance_px=80):
    if len(lanes_xy) == 0:
        return False
    x1, y1, x2, y2 = obj['bbox']
    sample_points = [
        ((x1 + x2) / 2.0, y2),
        (x1, y2),
        (x2, y2),
    ]
    for cx, cy in sample_points:
        lane_xs = []
        for lane_xy in lanes_xy:
            lx = interpolate_x_at_y(lane_xy, cy)
            if lx is not None:
                lane_xs.append(lx)
        if len(lane_xs) == 0:
            continue
        min_dist = min(abs(cx - lx) for lx in lane_xs)
        if min_dist <= distance_px:
            return True
    return False


def draw_front_objects(frame, objects, lanes_xy, names=None):
    for obj in objects:
        if 'bbox' not in obj:
            continue
        x1, y1, x2, y2 = obj['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(obj.get('conf', 0.0))
        in_lane = (
            is_point_in_lane((x1, y2), lanes_xy)
            or is_point_in_lane(((x1 + x2) / 2.0, y2), lanes_xy)
            or is_point_in_lane((x2, y2), lanes_xy)
        )

        # Only display if any of the 3 bottom lane-contact points are in lane.
        if not in_lane:
            continue

        color = (0, 0, 255)
        label = 'in_lane'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_ankle_point(frame, pt, color, label):
    if pt is None:
        return
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(frame, (x, y), 7, color, -1)
    cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

# --- MAIN THREAD: AI PROCESSING & DISPLAY ---
def main():
    global is_running, cv_ready, condition_since, last_auto_stop_send_ts
    print('[LAPTOP] Loading AI Models... Please wait.')
    device = choose_device(DEFAULT_DEVICE)
    cfg = Config.fromfile(resolve_path(DEFAULT_CONFIG))

    lane_model = build_net(cfg)
    load_checkpoint(lane_model, resolve_path(DEFAULT_CHECKPOINT), device)
    lane_model.to(device)
    lane_model.eval()

    YOLO = importlib.import_module('ultralytics').YOLO
    front_yolo = YOLO(resolve_path(DEFAULT_FRONT_YOLO))
    rear_pose = YOLO(resolve_path(DEFAULT_REAR_POSE))
    print(f'[LAPTOP] Models loaded on {device}. Ready!')

    # --- Synthetic warmup: force MPS kernel compilation before any real frame ---
    # This means the first real inference will be fast, not slow.
    print('[CV] Running synthetic warmup to compile MPS kernels...')
    try:
        dummy = torch.zeros(1, 3, cfg.img_h, cfg.img_w, device=device)
        with torch.inference_mode():
            out_dummy = lane_model({'img': dummy})
            _ = lane_model.heads.get_lanes(out_dummy)[0]
        print('[CV] ✅ Synthetic warmup complete — MPS kernels compiled')
    except Exception as e:
        print(f'[CV] ⚠️  Synthetic warmup failed (non-fatal): {e}')

    threading.Thread(target=run_tcp_server, daemon=True).start()
    threading.Thread(target=receive_rear_video, daemon=True).start()
    threading.Thread(target=receive_front_video, daemon=True).start()

    print('[CV] Waiting for first real front frame to confirm CV readiness...')

    frame_idx = 0
    cached_front_objects = []
    front_placeholder = make_placeholder_frame('Front AI Camera')
    rear_placeholder = make_placeholder_frame('Rear Backup Camera')

    print("[SERVER] 🖥️ Displaying video feeds. Press 'q' to quit.")

    while True:
        front = latest_front_frame
        rear = latest_rear_frame
        front_display = front_placeholder.copy() if front is None else front.copy()
        rear_display = rear_placeholder.copy() if rear is None else rear.copy()

        # --- CV warmup: run one inference as soon as first frame arrives ---
        if not cv_ready and front is not None:
            try:
                _, t_warmup = preprocess_frame(front, cfg, device)
                with torch.inference_mode():
                    out_w = lane_model({'img': t_warmup})
                    _ = lane_model.heads.get_lanes(out_w)[0]
                cv_ready = True
                cv_ready_event.set()
                print('[CV] ✅ CV is ready — robot can now be started from the app')
            except Exception as e:
                print(f'[CV] ⚠️  Warmup inference failed: {e}')

        if is_running:
            latest_angle_deg = 0.0
            front_stop_conditions = []
            rear_stop_conditions = []

            if front is not None:
                frame_idx += 1
                vis_front, t_front = preprocess_frame(front, cfg, device)
                
                with torch.inference_mode():
                    out_front = lane_model({'img': t_front})
                    lanes_front = lane_model.heads.get_lanes(out_front)[0]
                lanes_xy_front = draw_lanes(vis_front, lanes_front, cfg, line_width=4)

                if frame_idx % 2 == 0:
                    cached_front_objects = oD.get_objects(vis_front.copy(), front_yolo, conf_thres=0.3)

                draw_front_objects(
                    vis_front,
                    cached_front_objects,
                    lanes_xy_front,
                    front_yolo.names if hasattr(front_yolo, 'names') else None,
                )

                front_detection_output = build_front_detection(
                    cached_front_objects,
                    lanes_xy_front,
                    vis_front.shape,
                    front_yolo.names if hasattr(front_yolo, 'names') else None,
                    close_ratio=0.7,
                )
                if len(front_detection_output['danger']) > 0:
                    front_stop_conditions.append('If object is in lane')
                if len(lanes_xy_front) == 0:
                    front_stop_conditions.append('No lane Detected')
                    front_stop_conditions.append('Robot out of lane')
                
                steer_helper = SteeringHelper(lanes_xy_front, frame_width=vis_front.shape[1], n_samples=20, threshold=10)
                steer_angle = max(min(steer_helper.heading_angle, math.radians(45)), -math.radians(45))
                latest_angle_deg = round(math.degrees(steer_angle), 2)

                robot_status = 'OUT_OF_LANE' if len(lanes_xy_front) == 0 else 'NORMAL'
                if abs(steer_helper.heading_angle) >= math.radians(70):
                    robot_status = 'LARGE_ANGLE'
                    front_stop_conditions.append('Corner Angle too extreme')

                with state_lock:
                    latest_state['front'] = {
                        'robot_status': robot_status,
                        'ANGLE': f'ANGLE={latest_angle_deg:.2f}',
                        'object_detection': front_detection_output,
                        'stop_conditions': front_stop_conditions,
                    }
                if (vis_front.shape[1] != front.shape[1]) or (vis_front.shape[0] != front.shape[0]):
                    vis_front = cv2.resize(vis_front, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_LINEAR)
                front_display = vis_front

            if rear is not None:
                vis_rear, t_rear = preprocess_frame(rear, cfg, device)
                
                with torch.inference_mode():
                    out_rear = lane_model({'img': t_rear})
                    lanes_rear = lane_model.heads.get_lanes(out_rear)[0]
                lanes_xy_rear = draw_lanes(vis_rear, lanes_rear, cfg, line_width=4)

                left_ankle, right_ankle = pD.get_ankle(vis_rear.copy(), rear_pose)
                draw_ankle_point(vis_rear, left_ankle, (0, 255, 255), 'L ankle')
                draw_ankle_point(vis_rear, right_ankle, (255, 0, 255), 'R ankle')
                rear_status, _, _ = feet_status(left_ankle, right_ankle, lanes_xy_rear)
                rear_detection_output = build_rear_detection(rear_status)
                if rear_status == 'Left out':
                    rear_stop_conditions.append('Left foot out')
                elif rear_status == 'Right out':
                    rear_stop_conditions.append('Right foot out')
                elif rear_status == 'Both out':
                    rear_stop_conditions.append('Both feet out')
                elif rear_status == 'No feet detected':
                    rear_stop_conditions.append('No feet detected')
                if len(lanes_xy_rear) == 0:
                    rear_stop_conditions.append('No lane Detected')
                    rear_stop_conditions.append('Robot out of lane')

                with state_lock:
                    latest_state['rear'] = {
                        'status': rear_status,
                        'object_detection': rear_detection_output,
                        'stop_conditions': rear_stop_conditions,
                    }
                if (vis_rear.shape[1] != rear.shape[1]) or (vis_rear.shape[0] != rear.shape[0]):
                    vis_rear = cv2.resize(vis_rear, (rear.shape[1], rear.shape[0]), interpolation=cv2.INTER_LINEAR)
                rear_display = vis_rear

            combined_stop_conditions = front_stop_conditions + rear_stop_conditions

            # --- Per-condition hold-time evaluation ---
            # A condition must be continuously active for its hold duration before STOP fires.
            FRONT_HOLD = {
                'If object is in lane':     HOLD_OBJECT_IN_LANE_SEC,
                'Corner Angle too extreme': HOLD_CORNER_ANGLE_SEC,
                'No lane Detected':         HOLD_FRONT_NO_LANE_SEC,
                'Robot out of lane':        HOLD_FRONT_OUT_LANE_SEC,
            }
            REAR_HOLD = {
                'Left foot out':     HOLD_LEFT_FOOT_SEC,
                'Right foot out':    HOLD_RIGHT_FOOT_SEC,
                'Both feet out':     HOLD_BOTH_FEET_SEC,
                'No feet detected':  HOLD_NO_FEET_SEC,
                'No lane Detected':  HOLD_REAR_NO_LANE_SEC,
                'Robot out of lane': HOLD_REAR_OUT_LANE_SEC,
            }

            now = time.time()
            actuator_stop_conditions = []
            active_keys = set()

            for c, hold in FRONT_HOLD.items():
                if c in front_stop_conditions:
                    key = f'front:{c}'
                    active_keys.add(key)
                    if key not in condition_since:
                        condition_since[key] = now
                    elif (now - condition_since[key]) >= hold:
                        actuator_stop_conditions.append(c)

            for c, hold in REAR_HOLD.items():
                if c in rear_stop_conditions:
                    key = f'rear:{c}'
                    active_keys.add(key)
                    if key not in condition_since:
                        condition_since[key] = now
                    elif (now - condition_since[key]) >= hold:
                        actuator_stop_conditions.append(f'Rear: {c}')

            # Clear timers for conditions no longer active this frame
            for key in list(condition_since.keys()):
                if key not in active_keys:
                    del condition_since[key]

            if len(actuator_stop_conditions) > 0:
                now = time.time()
                if (now - last_auto_stop_send_ts) >= AUTO_STOP_COOLDOWN_SEC:
                    print(f"[AUTO-STOP] Hard stop triggered: {actuator_stop_conditions}")
                    forward_command_to_pi('STOP')
                    last_auto_stop_send_ts = now
                    with state_lock:
                        latest_state['auto_stop_reason'] = actuator_stop_conditions
                        latest_state['running'] = False
                    is_running = False
                    pi_started = False
                    broadcast_status(force=True)
                    with state_lock:
                        latest_state['auto_stop_reason'] = []

            # Always send steering updates; Arduino/Pi can decide what to do while STOP is active.
            if is_running:
                send_angle_to_pi(latest_angle_deg)

            # Push state at steady cadence while running (even if one stream is delayed)
            with state_lock:
                latest_state['running'] = is_running
            broadcast_status(force=False)

            cv2.putText(front_display, 'MODE: RUNNING CV', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(rear_display, 'MODE: RUNNING CV', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Front AI Camera", front_display)
            cv2.imshow("Rear Backup Camera", rear_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Idle preview mode: keep showing raw feeds on laptop even before START
            cv2.putText(front_display, 'MODE: IDLE (RAW)', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(rear_display, 'MODE: IDLE (RAW)', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Front AI Camera", front_display)
            cv2.imshow("Rear Backup Camera", rear_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()