"""TCP server to receive START/STOP commands from the mobile app and forward them to the Pi. Also handles WebSocket connections for real-time status updates."""
import json
import socket
import time
import cv2
import numpy as np
import Demo3.states.globals as g
from flask import request, jsonify
from Demo3.vision.helpers.steering_helper import angle_deg_to_servo_held
from Demo3.config.configs import *


def build_status_payload():
    """Construct the payload to send to WebSocket clients based on the latest state."""
    return {
        'running': g.latest_state.get('running', False),
        'cv_ready': g.cv_ready,
        'auto_stop_reason': g.latest_state.get('auto_stop_reason', []),
        'front': {
            'robot_status': g.latest_state.get('front', {}).get('robot_status', 'NORMAL'),
            'FRONT_ANGLE': g.latest_state.get('front', {}).get('FRONT_ANGLE', 'FRONT_ANGLE=90'),
            'object_detection': g.latest_state.get('front', {}).get('object_detection', {'warning': [], 'danger': []}),
            'stop_conditions': g.latest_state.get('front', {}).get('stop_conditions', []),
        },
        'rear': {
            'status': g.latest_state.get('rear', {}).get('status', 'No feet detected'),
            'REAR_ANGLE': g.latest_state.get('rear', {}).get('REAR_ANGLE', 'REAR_ANGLE=90'),
            'object_detection': g.latest_state.get('rear', {}).get('object_detection', {'warning': [], 'danger': []}),
            'stop_conditions': g.latest_state.get('rear', {}).get('stop_conditions', []),
        },
    }

def broadcast_status(force=False):
    """Send latest status to all connected WebSocket clients. If force=False, only send if running and rate-limit to WS_PUSH_HZ."""
    with g.state_lock:
        payload = build_status_payload()

    if (not force) and (not payload.get('running', False)):
        return

    now = time.time()
    if (not force) and ((now - g.last_ws_push_ts) < (1.0 / g.WS_PUSH_HZ)):
        return
    g.last_ws_push_ts = now

    serialized = json.dumps(payload)
    dead = []
    with g.ws_lock:
        for ws in g.ws_clients:
            try:
                ws.send(serialized)
            except Exception:
                dead.append(ws)
        for ws in dead:
            g.ws_clients.discard(ws)

def forward_command_to_pi(command):
    """Forward START/STOP command to all configured Pi IPs."""
    payload = f"{command}\n".encode('utf-8')
    for pi_ip in PI_IPS:
        print(f"[PI CMD] {command} -> {pi_ip}:{PI_CMD_PORT}")
        g.tx_socket.sendto(payload, (pi_ip, PI_CMD_PORT))

# --- NETWORKING THREADS ---
@g.app.route('/command', methods=['POST'])
def receive_command():

    command = (request.json or {}).get('action', '').strip().upper()

    if command == 'START':
        g.is_running = True
        g.pi_started = False
        g.condition_since = {}
        print('📱 APP SAYS START: CV started locally — Pi will be notified on first angle')
        with g.state_lock:
            g.latest_state['running'] = True
        broadcast_status(force=True)
        return jsonify({'status': 'success', 'running': True}), 200

    if command == 'STOP':
        g.is_running = False
        print('📱 APP SAYS STOP: Forwarding to Pi...')
        forward_command_to_pi('STOP')
        g.reset_steering_to_default()
        g.full_state_reset()
        with g.state_lock:
            g.latest_state['running'] = False
        broadcast_status(force=True)
        return jsonify({'status': 'success', 'running': False}), 200

    return jsonify({'status': 'error'}), 400


@g.app.route('/status', methods=['GET'])
def get_status():

    with g.state_lock:
        return jsonify(build_status_payload()), 200


@g.sock.route('/ws/status')
def ws_status(ws):
    with g.ws_lock:
        g.ws_clients.add(ws)
    try:
        with g.state_lock:
            ws.send(json.dumps(build_status_payload()))
        while True:
            msg = ws.receive()
            if msg is None:
                break
    finally:
        with g.ws_lock:
            g.ws_clients.discard(ws)

def run_tcp_server():
    g.app.run(host='0.0.0.0', port=API_PORT, debug=False, use_reloader=False)

def receive_rear_video():
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((HOST_IP, REAR_PORT))
    print(f"[SERVER] 🟢 Listening for REAR camera on port {REAR_PORT}")
    while True:
        try:
            data, _ = udp_sock.recvfrom(MAX_DGRAM)
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                g.latest_rear_frame = frame
                g.rear_frame_ts = time.time()
        except Exception:
            pass

def receive_front_video():
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((HOST_IP, FRONT_PORT))
    print(f"[SERVER] 🔵 Listening for FRONT camera on port {FRONT_PORT}")
    while True:
        try:
            data, _ = udp_sock.recvfrom(MAX_DGRAM)
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                g.latest_front_frame = frame
                g.front_frame_ts = time.time()
        except Exception:
            pass

def send_angles_to_pi(angle_front, rear_servo=90):
    # On the very first angle of a run, send START to Pi first so it is ready to move.
    # Guard with is_running to avoid a race where STOP arrives mid-loop.
    if not g.pi_started:
        if not g.is_running:
            return
        print('[PI CMD] First angle computed — sending START to Pi now')
        forward_command_to_pi('START')
        g.pi_started = True

    front_servo = angle_deg_to_servo_held(angle_front)
    if front_servo is None:
        return  # Angle hasn't been stable for STEER_HOLD_SEC yet — skip update

    body = f"FRONT_ANGLE={front_servo},REAR_ANGLE={rear_servo}"
    for pi_ip in PI_IPS:
        print(f"[PI STEER] {body} (front={angle_front:.2f}°, rear_servo={rear_servo}) -> {pi_ip}:{PI_CMD_PORT}")
        g.tx_socket.sendto(body.encode('utf-8'), (pi_ip, PI_CMD_PORT))
