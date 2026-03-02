import os
import sys
import cv2
import time
import torch
import socket
import importlib
import threading
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_flask = importlib.import_module('flask')
Flask = _flask.Flask
request = _flask.request
jsonify = _flask.jsonify

from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from tools.YOLOv8n.front_camera import FrontCamera
import tools.YOLOv8n.poseDetector as pD

HOST_IP = '0.0.0.0'
FRONT_PORT = 8000
REAR_PORT = 8002
MAX_DGRAM = 65507

PI_IP = ''  # set your Pi IP
PI_CMD_PORT = 8001
PI_STATUS_PORT = 8003

DEFAULT_CONFIG = 'configs/clrnet/clr_resnet18_tusimple.py'
DEFAULT_CHECKPOINT = 'checkpoints/tusimple_r18.pth'
DEFAULT_DEVICE = 'mps'
DEFAULT_FRONT_YOLO = 'checkpoints/yolov8n_int8.tflite'
DEFAULT_REAR_POSE = 'checkpoints/yolov8n-pose_int8.tflite'
DEFAULT_FRONT_CLOSE_RATIO = 0.7
DEFAULT_FRONT_YOLO_STRIDE = 2
DEFAULT_LOG_EVERY = 30


def resolve_path(path):
    if os.path.isabs(path):
        return path
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    root_candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_candidate):
        return root_candidate
    return cwd_candidate


def choose_device(device_flag):
    if device_flag == 'cuda':
        return torch.device('cuda')
    if device_flag == 'mps':
        return torch.device('mps')
    if device_flag == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Rear-camera helpers (kept intact)
# ---------------------------------------------------------------------------

def normalize_point(pt):
    if pt is None:
        return None
    return float(pt[0]), float(pt[1])


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
    for i in range(len(xs_at_y) - 1):
        if xs_at_y[i] <= cx <= xs_at_y[i + 1]:
            return True
    return False


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


def draw_lanes(frame, lanes, cfg, line_width=4):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    lanes_xy = extract_lane_xy(lanes, cfg, frame.shape)
    for i, xy in enumerate(lanes_xy):
        color = colors[i % len(colors)]
        for j in range(1, len(xy)):
            cv2.line(frame, xy[j - 1], xy[j], color, thickness=line_width)
    return lanes_xy


def preprocess_frame(frame, cfg, device):
    frame = cv2.resize(frame, (cfg.ori_img_w, cfg.ori_img_h), interpolation=cv2.INTER_LINEAR)
    cropped = frame[cfg.cut_height:, :, :]
    resized = cv2.resize(cropped, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return frame, tensor


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        cleaned[k[len('module.'):]] = v if k.startswith('module.') else v
    model.load_state_dict(cleaned, strict=False)


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


def draw_foot(frame, pt, color, label):
    if pt is None:
        return
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
    cv2.circle(frame, (x, y), 4, color, -1)
    cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# UDP receiver
# ---------------------------------------------------------------------------

class UDPReceiver:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.settimeout(0.2)
        self.data = None
        self.lock = threading.Lock()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                packet, _ = self.sock.recvfrom(MAX_DGRAM)
                with self.lock:
                    self.data = packet
            except socket.timeout:
                continue
            except Exception:
                if not self.stopped:
                    self.stopped = True

    def get_latest_frame(self):
        with self.lock:
            return self.data

    def stop(self):
        self.stopped = True
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Flask API
# ---------------------------------------------------------------------------

app = Flask(__name__)
tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

is_running = False
front_receiver = None
rear_receiver = None

state_lock = threading.Lock()
latest_state = {
    'running': False,
    'front': {'warning': [], 'danger': []},
    'rear': {'status': 'No feet detected'}
}


def send_to_pi(message, port):
    if not PI_IP:
        return
    try:
        tx_socket.sendto(message.encode('utf-8'), (PI_IP, port))
    except Exception:
        pass


@app.route('/command', methods=['POST'])
def receive_command():
    global is_running, front_receiver, rear_receiver

    command = (request.json or {}).get('action', '').strip().upper()
    if command not in ('START', 'STOP'):
        return jsonify({'status': 'error', 'message': 'action must be START or STOP'}), 400

    send_to_pi(command, PI_CMD_PORT)

    if command == 'START' and not is_running:
        is_running = True
        front_receiver = UDPReceiver(HOST_IP, FRONT_PORT).start()
        rear_receiver = UDPReceiver(HOST_IP, REAR_PORT).start()
    elif command == 'STOP' and is_running:
        is_running = False
        if front_receiver:
            front_receiver.stop()
            front_receiver = None
        if rear_receiver:
            rear_receiver.stop()
            rear_receiver = None

    with state_lock:
        latest_state['running'] = is_running

    return jsonify({'status': 'success', 'running': is_running}), 200


@app.route('/status', methods=['GET'])
def get_status():
    with state_lock:
        return jsonify(latest_state), 200


def run_flask_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


def decode_packet(packet):
    if packet is None:
        return None
    np_data = np.frombuffer(packet, dtype=np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    global is_running

    threading.Thread(target=run_flask_server, daemon=True).start()
    print('[LAPTOP] Control API: http://0.0.0.0:5000 (POST /command, GET /status)')

    print('[LAPTOP] Loading models...')

    # --- Front camera: delegate entirely to FrontCamera ---
    front_cam = FrontCamera(
        config=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        device=DEFAULT_DEVICE,
        yolo_model=DEFAULT_FRONT_YOLO,
        close_ratio=DEFAULT_FRONT_CLOSE_RATIO,
        yolo_stride=DEFAULT_FRONT_YOLO_STRIDE,
        log_interval=0.0,          # we handle our own periodic log below
        visualization=True,
        source='0',                # source is not used here; we feed frames manually
        no_objects=False,
    )

    # --- Rear camera: keep existing lane model + pose model ---
    device = choose_device(DEFAULT_DEVICE)
    cfg = Config.fromfile(resolve_path(DEFAULT_CONFIG))
    from clrnet.models.registry import build_net
    lane_model = build_net(cfg)
    load_checkpoint(lane_model, resolve_path(DEFAULT_CHECKPOINT), device)
    lane_model.to(device)
    lane_model.eval()

    YOLO = importlib.import_module('ultralytics').YOLO
    rear_pose = YOLO(resolve_path(DEFAULT_REAR_POSE))

    print(f'[LAPTOP] Ready. Waiting for START...')

    last_rear_status = None
    frame_idx = 0
    log_every = max(0, int(DEFAULT_LOG_EVERY))

    try:
        while True:
            if not is_running or front_receiver is None or rear_receiver is None:
                cv2.destroyAllWindows()
                time.sleep(0.05)
                continue

            front_packet = front_receiver.get_latest_frame()
            rear_packet = rear_receiver.get_latest_frame()

            # ----------------------------------------------------------------
            # Front camera — processed entirely via FrontCamera
            # ----------------------------------------------------------------
            front_raw = decode_packet(front_packet)
            if front_raw is not None:
                frame_idx += 1
                front_cam.process(front_raw)

                # Read results from FrontCamera attributes
                detection_output = front_cam.get_object_detection_output()
                # warning_names = detection_output['warning']
                # danger_names  = detection_output['danger']

                # Periodic console log
                # TODO: consider removing this part if it is printed in the process()
                # if log_every and (frame_idx % log_every == 0):
                #     print(f'[FRONT] Robot Status: {front_cam.robot_stat.name}, '
                #           f'Steering Angle: {front_cam.steer_angle and round(__import__("math").degrees(front_cam.steer_angle), 2)}°')
                #     print(f'[FRONT] Object detection: {detection_output}')
                #     stop_conds = front_cam.get_stop_conditions(time.time())
                #     if stop_conds:
                #         print(f'[FRONT] Stop conditions: {stop_conds}')
                #
                # with state_lock:
                #     latest_state['front'] = {
                #         'warning': warning_names,
                #         'danger':  danger_names,
                #     }

                cv2.imshow('Front CV', front_cam.frame)

            # ----------------------------------------------------------------
            # Rear camera — unchanged logic
            # ----------------------------------------------------------------
            rear_raw = decode_packet(rear_packet)
            if rear_raw is not None:
                vis_rear, t_rear = preprocess_frame(rear_raw, cfg, device)
                with torch.inference_mode():
                    out_rear = lane_model({'img': t_rear})
                    lanes_rear = lane_model.heads.get_lanes(out_rear)[0]

                lanes_xy_rear = draw_lanes(vis_rear, lanes_rear, cfg, line_width=4)
                left_ankle, right_ankle = pD.get_ankle(vis_rear.copy(), rear_pose)
                status, left_in, right_in = feet_status(left_ankle, right_ankle, lanes_xy_rear)

                left_color  = (0, 255, 0) if left_in  else (0, 0, 255)
                right_color = (0, 255, 0) if right_in else (0, 0, 255)
                status_color = (0, 255, 0) if status == 'Safe' else (0, 0, 255)

                draw_foot(vis_rear, normalize_point(left_ankle),  left_color,  'L')
                draw_foot(vis_rear, normalize_point(right_ankle), right_color, 'R')
                cv2.putText(vis_rear, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, status_color, 2, cv2.LINE_AA)

                if status != last_rear_status:
                    print(f'[REAR] {status}')
                    send_to_pi(f'REAR_STATUS:{status}', PI_STATUS_PORT)
                last_rear_status = status

                with state_lock:
                    latest_state['rear'] = {'status': status}

                cv2.imshow('Rear CV', vis_rear)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if front_receiver:
            front_receiver.stop()
        if rear_receiver:
            rear_receiver.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

