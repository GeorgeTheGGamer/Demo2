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
from collections import deque
import numpy as np
from flask import Flask, request, jsonify
from flask_sock import Sock

from tools.helper.focus_helper import FocusHelper

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- AI & AUTONOMY IMPORTS ---
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
import objectDetector as oD
import poseDetector as pD
from tools.helper.steering_helper import SteeringHelper
from tools.helper.lane_fixer import LaneFixer

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
MINMAX_ANGLE = 45.0  # Max heading angle in degrees for servo mapping
STRAIGHT_THRESHOLD = 5
STEER_THRESHOLD = 10

# --- GLOBALS ---
latest_rear_frame = None
latest_front_frame = None
is_running = False
cv_ready = False
cv_ready_event = threading.Event()
pi_started = False
condition_since = {}   # { 'front:<cond>' | 'rear:<cond>' : first_seen_timestamp }
state = 'STRAIGHT'

# Steering hold state: tracks the current candidate servo value and when it was first seen
_steer_candidate = {'servo': None, 'since': 0.0}

# Rear servo state: tracks last committed servo value and last time feet were detected
_rear_servo_state = {'servo': 90, 'last_seen': 0.0}

# Frame timestamps — updated by receive threads; used for staleness checks
front_frame_ts = 0.0
rear_frame_ts  = 0.0

tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
app = Flask(__name__)
sock = Sock(app)

state_lock = threading.Lock()
ws_lock = threading.Lock()
ws_clients = set()
last_ws_push_ts = 0.0
WS_PUSH_HZ = 5.0

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

# How long (seconds) the heading angle must remain within the same servo
# threshold band before the servo is actually updated. Prevents jitter on
# borderline angles and makes turning smoother.
STEER_HOLD_SEC = 3.0

# How long (seconds) with no feet detected before the rear servo resets to 90° (centre).
REAR_NO_FEET_HOLD_SEC = 5.0

# Max age (seconds) of a camera frame before it is treated as stale / no stream.
FRAME_MAX_AGE_SEC = 6.0

# Majority-vote steering: number of recent frames to consider and minimum
# agreement count required before a new servo value is committed.
# EMA smooths the raw angle first, then majority-vote confirms direction.
# E.g. window=5, threshold=3 means 3/5 EMA-smoothed frames must agree.
STEER_VOTE_WINDOW    = 5    # rolling window size (reduced for low-FPS responsiveness)
STEER_VOTE_THRESHOLD = 3    # minimum votes needed (out of STEER_VOTE_WINDOW)
STEER_EMA_ALPHA      = 0.4  # EMA smoothing factor: 0=no update, 1=no smoothing. Tune during testing.

# Rolling window of recent servo candidates for majority-vote steering
_angle_window: deque = deque(maxlen=STEER_VOTE_WINDOW)
# EMA state for raw angle smoothing (seeded on first frame)
_ema_angle = None

latest_state = {
    'running': False,
    'auto_stop_reason': [],
    'front': {
        'robot_status': 'NORMAL',
        'FRONT_ANGLE': 'FRONT_ANGLE=90',
        'object_detection': {'warning': [], 'danger': []},
        'stop_conditions': [],
    },
    'rear': {
        'status': 'No feet detected',
        'REAR_ANGLE': 'REAR_ANGLE=90',
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

def draw_lanes(frame, lanes_xy, line_width=4):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    for i, xy in enumerate(lanes_xy):
        if len(xy) < 2:
            continue
        color = colors[i % len(colors)]
        for j in range(1, len(xy)):
            cv2.line(frame, xy[j - 1], xy[j], color, thickness=line_width)

def extract_lane_xy(lanes, cfg, frame_shape):
    lanes_xy = []
    h, w = frame_shape[:2]
    for lane in lanes:
        pts = lane.to_array(cfg)
        xy = []
        for p in pts:
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h:
                xy.append((x, y))
        if len(xy) >= 2:
            lanes_xy.append(xy)
    lanes_xy.sort(key=lambda xys: xys[0][0])
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
            'FRONT_ANGLE': latest_state.get('front', {}).get('FRONT_ANGLE', 'FRONT_ANGLE=90'),
            'object_detection': latest_state.get('front', {}).get('object_detection', {'warning': [], 'danger': []}),
            'stop_conditions': latest_state.get('front', {}).get('stop_conditions', []),
        },
        'rear': {
            'status': latest_state.get('rear', {}).get('status', 'No feet detected'),
            'REAR_ANGLE': latest_state.get('rear', {}).get('REAR_ANGLE', 'REAR_ANGLE=90'),
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


def reset_steering_to_default():
    """Send both servos back to 90° (straight) and clear steering hold-timer state."""
    global _steer_candidate, _rear_servo_state, _angle_window, _ema_angle
    _steer_candidate = {'servo': None, 'since': 0.0}
    _rear_servo_state = {'servo': 90, 'last_seen': 0.0}
    _angle_window.clear()
    _ema_angle = None
    body = "FRONT_ANGLE=90,REAR_ANGLE=90"
    for _ in range(3):  # send 3 times to survive UDP packet drops
        for pi_ip in PI_IPS:
            print(f"[PI STEER] RESET -> {body} -> {pi_ip}:{PI_CMD_PORT}")
            tx_socket.sendto(body.encode('utf-8'), (pi_ip, PI_CMD_PORT))
        time.sleep(0.05)


def full_state_reset():
    """
    Full reset of all runtime state. Called on STOP (manual or auto) so the
    next START is completely independent — no stale angles, conditions, or timers.
    """
    global _steer_candidate, _rear_servo_state, _angle_window, _ema_angle
    global condition_since, latest_front_frame, latest_rear_frame
    global front_frame_ts, rear_frame_ts, pi_started

    # Steering
    _steer_candidate  = {'servo': None, 'since': 0.0}
    _rear_servo_state = {'servo': 90, 'last_seen': 0.0}
    _angle_window.clear()
    _ema_angle = None

    # Stop condition hold timers
    condition_since.clear()

    # Drop stale frames so next run starts fresh
    latest_front_frame = None
    latest_rear_frame  = None
    front_frame_ts     = 0.0
    rear_frame_ts      = 0.0

    # Pi handshake — force re-START on next run
    pi_started = False


# --- NETWORKING THREADS ---
@app.route('/command', methods=['POST'])
def receive_command():
    global is_running
    command = (request.json or {}).get('action', '').strip().upper()

    if command == 'START':
        global pi_started, condition_since
        is_running = True
        pi_started = False
        condition_since = {}
        print('📱 APP SAYS START: CV started locally — Pi will be notified on first angle')
        with state_lock:
            latest_state['running'] = True
        broadcast_status(force=True)
        return jsonify({'status': 'success', 'running': True}), 200

    if command == 'STOP':
        is_running = False
        print('📱 APP SAYS STOP: Forwarding to Pi...')
        forward_command_to_pi('STOP')
        reset_steering_to_default()
        full_state_reset()
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
    global latest_rear_frame, rear_frame_ts
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST_IP, REAR_PORT))
    print(f"[SERVER] 🟢 Listening for REAR camera on port {REAR_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(MAX_DGRAM)
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                latest_rear_frame = frame
                rear_frame_ts = time.time()
        except Exception as e:
            pass

def receive_front_video():
    global latest_front_frame, front_frame_ts
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST_IP, FRONT_PORT))
    print(f"[SERVER] 🔵 Listening for FRONT camera on port {FRONT_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(MAX_DGRAM)
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                latest_front_frame = frame
                front_frame_ts = time.time()
        except Exception as e:
            pass

def angle_deg_to_servo(angle_deg: float) -> int:
    """
    Map heading angle (-45 to +45 degrees) to fixed servo constants (70-110).
    Positive angle = turn right -> higher servo value (towards 110)
    Negative angle = turn left  -> lower servo value  (towards 70)
    90 = straight ahead. Dead-band: ±10° to account for angle inaccuracy.
    Fixed constants: 70, 75, 80, 85, 90, 95, 100, 105, 110 (5° increments)
    """
    a = angle_deg
    if a >= 38:
        return 110      # Hard Right
    elif a >= 28:
        return 105      # Moderate Right
    elif a >= 18:
        return 100      # Slight Right
    elif a >= 10:
        return 95       # Nudge Right
    elif a >= -10:
        return 90       # Straight  ← ±10° dead-band
    elif a >= -17:
        return 85       # Nudge Left
    elif a >= -27:
        return 80       # Slight Left
    elif a >= -37:
        return 75       # Moderate Left
    else:
        return 70       # Hard Left


def angle_deg_to_servo_held(angle_deg: float) -> int | None:
    """
    Two-stage filter:
    1. EMA smooths the raw angle first (removes noisy/spiked CLRNet readings)
    2. Majority-vote on the EMA-smoothed servo value (confirms direction)
    This ensures only angles that reliably reflect the true lane direction are sent.
    Returns None until window is full or no consensus is reached.
    """
    global _angle_window, _ema_angle

    # Stage 1: EMA smooth the raw angle
    if _ema_angle is None:
        _ema_angle = angle_deg  # seed on first frame
        return None
    _ema_angle = STEER_EMA_ALPHA * angle_deg + (1.0 - STEER_EMA_ALPHA) * _ema_angle

    # Stage 2: Map smoothed angle to servo, then majority-vote
    candidate = angle_deg_to_servo(_ema_angle)
    _angle_window.append(candidate)

    if len(_angle_window) < STEER_VOTE_WINDOW:
        return None  # window not yet full

    majority = max(set(_angle_window), key=_angle_window.count)
    if _angle_window.count(majority) >= STEER_VOTE_THRESHOLD:
        return majority

    return None  # no consensus yet


def rear_angle_to_servo(angle_deg: float) -> int:
    """
    Map rear heading angle (-45 to +45 degrees) to a servo value in 5-degree
    increments between 45 and 135.
    Positive angle = person to the right -> higher servo (towards 135)
    Negative angle = person to the left  -> lower servo  (towards 45)
    90 = centred
    """
    clamped = max(min(angle_deg, 45.0), -45.0)
    raw = 90.0 + clamped
    snapped = round(raw / 5.0) * 5
    return max(45, min(135, snapped))


def get_rear_servo(angle_deg, feet_detected: bool) -> int:
    """
    Returns the rear servo value (45-135 in 5° increments).
    - If feet are detected: compute from angle and update last-seen timestamp.
    - If no feet: hold last position. After REAR_NO_FEET_HOLD_SEC seconds,
      reset to 90° (centre).
    """
    global _rear_servo_state
    now = time.time()

    if feet_detected:
        servo = rear_angle_to_servo(angle_deg if angle_deg is not None else 0.0)
        _rear_servo_state['servo'] = servo
        _rear_servo_state['last_seen'] = now
        return servo
    else:
        # No feet — hold last position until timeout, then centre
        if (now - _rear_servo_state['last_seen']) >= REAR_NO_FEET_HOLD_SEC:
            _rear_servo_state['servo'] = 90
        return _rear_servo_state['servo']


def send_angles_to_pi(angle_front, rear_servo=90):
    global pi_started
    # On the very first angle of a run, send START to Pi first so it is ready to move.
    # Guard with is_running to avoid a race where STOP arrives mid-loop.
    if not pi_started:
        if not is_running:
            return
        print('[PI CMD] First angle computed — sending START to Pi now')
        forward_command_to_pi('START')
        pi_started = True

    front_servo = angle_deg_to_servo_held(angle_front)
    if front_servo is None:
        return  # Angle hasn't been stable for STEER_HOLD_SEC yet — skip update

    body = f"FRONT_ANGLE={front_servo},REAR_ANGLE={rear_servo}"
    for pi_ip in PI_IPS:
        print(f"[PI STEER] {body} (front={angle_front:.2f}°, rear_servo={rear_servo}) -> {pi_ip}:{PI_CMD_PORT}")
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

def calculate_midpoint(left_ankle, right_ankle):
    """
    Calculate the midpoint between left and right ankles.
    :return: midpoint coordinate [x, y] or None if either ankle is None
    """
    if left_ankle is None or right_ankle is None:
        return None

    x_mid = (left_ankle[0] + right_ankle[0]) / 2
    y_mid = (left_ankle[1] + right_ankle[1]) / 2

    return [x_mid, y_mid]

def calculate_angle_to_center(midpoint, frame):
    frame_height, frame_width = frame.shape[:2]
    if midpoint is None:
        return None

    # Vector from midpoint to top-center (frame_width/2, 0)
    center_x = frame_width / 2
    dx = midpoint[0] - center_x
    dy = 0 - midpoint[1]
    angle_x = np.degrees(np.arctan2(dx, -dy))

    return angle_x

def visualize_angle(frame, angle):
    if angle is None:
        return
    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h - 50
    length = 100
    end_x = int(center_x + length * np.sin(np.radians(angle)))
    end_y = int(center_y - length * np.cos(np.radians(angle)))
    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (255, 0, 255), 3)
    cv2.putText(frame,
                f'Angle: {angle:.1f} deg',
                (center_x - 60, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
                cv2.LINE_AA)

# --- MAIN THREAD: AI PROCESSING & DISPLAY ---
def main():
    global is_running, cv_ready, condition_since
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

    # Init helpers
    ankle_focus = FocusHelper()
    front_fixer = LaneFixer()
    rear_fixer = LaneFixer()

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
        now_ts = time.time()
        front = latest_front_frame if (front_frame_ts > 0 and (now_ts - front_frame_ts) <= FRAME_MAX_AGE_SEC) else None
        rear  = latest_rear_frame  if (rear_frame_ts  > 0 and (now_ts - rear_frame_ts)  <= FRAME_MAX_AGE_SEC) else None
        front_display = front_placeholder.copy() if front is None else front.copy()
        rear_display  = rear_placeholder.copy()  if rear  is None else rear.copy()

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

        # Initialise per-frame condition lists (also used by auto-resume check in else block)
        front_stop_conditions = []
        rear_stop_conditions  = []

        if is_running:
            latest_angle_deg = 0.0
            rear_servo_val = 90

            if front is not None:
                frame_idx += 1
                vis_front, t_front = preprocess_frame(front, cfg, device)
                
                with torch.inference_mode():
                    out_front = lane_model({'img': t_front})
                    lanes_front = lane_model.heads.get_lanes(out_front)[0]

                # Fix lanes and update lanes
                lanes_xy_front = extract_lane_xy(lanes_front, cfg, vis_front.shape)
                lanes_xy_front = front_fixer.fix(lanes_xy_front, frame_width=vis_front.shape[1])
                draw_lanes(vis_front, lanes_xy_front)

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

                steer_helper = None
                # If current state is STRAIGHT, set threshold to STRAIGHT_THRESHOLD to make robot go straight.
                if state == 'STRAIGHT':
                    steer_helper = SteeringHelper(lanes_xy_front, vis_front.shape[:2], n_samples=20, threshold=STRAIGHT_THRESHOLD)

                # If current state is LEFT or RIGHT, use the normal STEER_THRESHOLD to allow sharper turns.
                if state == 'LEFT' or state == 'RIGHT':
                    steer_helper = SteeringHelper(lanes_xy_front, vis_front.shape[:2], n_samples=20, threshold=STEER_THRESHOLD)

                steer_angle = max(min(steer_helper.heading_angle, math.radians(MINMAX_ANGLE)), -math.radians(MINMAX_ANGLE))
                latest_angle_deg = round(math.degrees(steer_angle), 2)

                robot_status = 'OUT_OF_LANE' if len(lanes_xy_front) == 0 else 'NORMAL'
                if abs(steer_helper.heading_angle) >= math.radians(70):
                    robot_status = 'LARGE_ANGLE'
                    front_stop_conditions.append('Corner Angle too extreme')

                servo_val = angle_deg_to_servo(latest_angle_deg)
                with state_lock:
                    latest_state['front'] = {
                        'robot_status': robot_status,
                        'FRONT_ANGLE': f'FRONT_ANGLE={servo_val}',
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

                # Fix lanes and update lanes
                lanes_xy_rear = extract_lane_xy(lanes_rear, cfg, vis_rear.shape)
                lanes_xy_rear = rear_fixer.fix(lanes_xy_rear, frame_width=vis_rear.shape[1])
                draw_lanes(vis_rear, lanes_xy_rear)

                # left_ankle, right_ankle = pD.get_ankle(vis_rear.copy(), rear_pose)
                # Get the focused ankle points using the FocusHelper, which may provide more stable detections by focusing on the expected person
                ankles = pD.get_ankles(vis_rear.copy(), rear_pose)
                ankle_focus.update_frame_size(vis_rear.shape[1], vis_rear.shape[0])
                left_ankle, right_ankle = ankle_focus.focus(ankles)
                draw_ankle_point(vis_rear, left_ankle, (0, 255, 255), 'L ankle')
                draw_ankle_point(vis_rear, right_ankle, (255, 0, 255), 'R ankle')

                # Calculate midpoint and angle to center for rear display and potential future use
                midpoint = calculate_midpoint(left_ankle, right_ankle)
                draw_ankle_point(vis_rear, normalize_point(midpoint), (255, 255, 0), 'M')
                angle_rear = calculate_angle_to_center(midpoint, vis_rear)
                visualize_angle(vis_rear, angle_rear)

                feet_detected = midpoint is not None
                rear_servo_val = get_rear_servo(angle_rear, feet_detected)

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
                        'REAR_ANGLE': f'REAR_ANGLE={rear_servo_val}',
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
                print(f"[AUTO-STOP] Hard stop triggered: {actuator_stop_conditions}")
                forward_command_to_pi('STOP')
                reset_steering_to_default()
                full_state_reset()
                with state_lock:
                    latest_state['auto_stop_reason'] = actuator_stop_conditions
                    latest_state['running'] = False
                is_running = False
                broadcast_status(force=True)
                with state_lock:
                    latest_state['auto_stop_reason'] = []

            # Always send steering updates; Arduino/Pi can decide what to do while STOP is active.
            if is_running:
                send_angles_to_pi(latest_angle_deg, rear_servo_val)

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
            cv2.putText(rear_display,  'MODE: IDLE (RAW)', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Front AI Camera", front_display)
            cv2.imshow("Rear Backup Camera", rear_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()