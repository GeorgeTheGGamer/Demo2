"""Store global variables and provide helper functions for managing state, WebSocket clients, and Pi communication."""

import threading
import socket
import time
from collections import deque
from flask import Flask
from flask_sock import Sock

from Demo3.config.configs import *  # Import all config parameters


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
steer_candidate = {'servo': None, 'since': 0.0}

# Rear servo state: tracks last committed servo value and last time feet were detected
rear_servo_state = {'servo': 90, 'last_seen': 0.0}

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

# Rolling window of recent servo candidates for growing-trend steering
angle_window: deque = deque(maxlen=STEER_VOTE_WINDOW)

# Dynamic steering threshold (degrees): angle is snapped to the nearest multiple of this value.
current_threshold = STRAIGHT_THRESHOLD
front_current_angle = 90.0
rear_current_angle = 90.0

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

def full_state_reset():
    """
    Full reset of all runtime state. Called on STOP (manual or auto) so the
    next START is completely independent — no stale angles, conditions, or timers.
    """
    global steer_candidate, rear_servo_state, angle_window
    global condition_since, latest_front_frame, latest_rear_frame
    global front_frame_ts, rear_frame_ts, pi_started

    # Steering
    steer_candidate  = {'servo': None, 'since': 0.0}
    rear_servo_state = {'servo': 90, 'last_seen': 0.0}
    angle_window.clear()

    # Stop condition hold timers
    condition_since.clear()

    # Drop stale frames so next run starts fresh
    latest_front_frame = None
    latest_rear_frame  = None
    front_frame_ts     = 0.0
    rear_frame_ts      = 0.0

    # Pi handshake — force re-START on next run
    pi_started = False

def reset_steering_to_default():
    """Send both servos back to 90° (straight) and clear steering hold-timer state."""
    global steer_candidate, rear_servo_state, angle_window
    steer_candidate  = {'servo': None, 'since': 0.0}
    rear_servo_state = {'servo': 90, 'last_seen': 0.0}
    angle_window.clear()
    body = "FRONT_ANGLE=90,REAR_ANGLE=90"
    for _ in range(3):  # send 3 times to survive UDP packet drops
        for pi_ip in PI_IPS:
            print(f"[PI STEER] RESET -> {body} -> {pi_ip}:{PI_CMD_PORT}")
            tx_socket.sendto(body.encode('utf-8'), (pi_ip, PI_CMD_PORT))
        time.sleep(0.05)
