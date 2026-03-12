"""TCP server to receive START/STOP commands from the mobile app and forward them to the Pi. Also handles WebSocket connections for real-time status updates."""
import json

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_sock import Sock
from Demo3.states.globals import *
from Demo3.vision.helpers.steering_helper import angle_deg_to_servo_held


def build_status_payload():
    """Construct the payload to send to WebSocket clients based on the latest state. This includes:"""
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
    """Send latest status to all connected WebSocket clients. If force=False, only send if running and rate-limit to WS_PUSH_HZ."""
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