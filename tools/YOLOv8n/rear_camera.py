import argparse
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import importlib

from clrnet.models.registry import build_net
from clrnet.utils.config import Config
import poseDetector as pD


# -----------------------------
# Local run defaults (edit here)
# -----------------------------
DEFAULT_CONFIG = 'configs/clrnet/clr_resnet18_tusimple.py'
DEFAULT_CHECKPOINT = 'checkpoints/tusimple_r18.pth'
DEFAULT_SOURCE = '0'
DEFAULT_DEVICE = 'auto'  # 'auto' | 'cuda' | 'mps' | 'cpu'
DEFAULT_POSE_MODEL = 'checkpoints/yolov8n-pose_int8.tflite'
LOG_INTERVAL_SECONDS = 5.0
FOOT_OUT_STOP_SECONDS = 3.0
NO_FEET_STOP_SECONDS = 5.0
REAR_OUT_OF_LANE_STOP_SECONDS = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description='CLRNet live demo (webcam/video)')
    parser.add_argument('config', nargs='?', default=DEFAULT_CONFIG,
                        help='config file path')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help='checkpoint file (.pth)')
    parser.add_argument('--source', default=DEFAULT_SOURCE,
                        help='camera index (e.g. 0) or video file path')
    parser.add_argument('--device', default=DEFAULT_DEVICE, choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='inference device')
    parser.add_argument('--conf', type=float, default=None,
                        help='override confidence threshold from config')
    parser.add_argument('--max-lanes', type=int, default=None,
                        help='override maximum lanes shown')
    parser.add_argument('--line-width', type=int, default=4)
    parser.add_argument('--no-pose', action='store_true',
                        help='disable pose/feet overlay')
    parser.add_argument('--pose-model', default=DEFAULT_POSE_MODEL,
                        help='YOLO pose model path (.pt/.tflite) relative to project root or absolute')
    parser.add_argument('--output', default=None,
                        help='optional output video path (e.g. demo.mp4)')
    parser.add_argument('--log-interval', type=float, default=LOG_INTERVAL_SECONDS,
                        help='print status every N seconds (0 disables periodic status logs)')
    return parser.parse_args()


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


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        if k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)


def preprocess_frame(frame, cfg, device):
    frame = cv2.resize(frame, (cfg.ori_img_w, cfg.ori_img_h), interpolation=cv2.INTER_LINEAR)
    cropped = frame[cfg.cut_height:, :, :]
    resized = cv2.resize(cropped, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return frame, tensor


def draw_lanes(frame, lanes, cfg, line_width=4):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    lanes_xy = []
    for lane in lanes:
        pts = lane.to_array(cfg)
        xy = []
        for p in pts:
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                xy.append((x, y))
        if len(xy) >= 2:
            lanes_xy.append(xy)

    lanes_xy.sort(key=lambda xys: xys[0][0])
    for i, xy in enumerate(lanes_xy):
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


def is_center_within_lane(center, lanes_xy):
    if len(lanes_xy) < 2:
        return False

    cx, cy = center
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


def draw_foot(frame, pt, color, label):
    if pt is None:
        return
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
    cv2.circle(frame, (x, y), 4, color, -1)
    cv2.putText(frame,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA)


def feet_status(left_ankle, right_ankle, lanes_xy):
    left = normalize_point(left_ankle)
    right = normalize_point(right_ankle)

    left_in = False if left is None else is_center_within_lane(left, lanes_xy)
    right_in = False if right is None else is_center_within_lane(right, lanes_xy)

    if left is None and right is None:
        return 'No feet detected', left_in, right_in
    if left_in and right_in:
        return 'Safe', left_in, right_in
    if (not left_in) and right_in:
        return 'Left out', left_in, right_in
    if left_in and (not right_in):
        return 'Right out', left_in, right_in
    return 'Both out', left_in, right_in


def rear_status_explanation(status):
    if status == 'No feet detected':
        return 'YOLO-Pose found a person but failed to identify ankle points.'
    if status == 'Safe':
        return 'Both left and right ankle coordinates are between the rear lane lines.'
    if status in ('Left out', 'Right out'):
        return 'One specific ankle is outside the detected lane boundary.'
    if status == 'Both out':
        return 'Both ankles are detected but neither is within the lane lines.'
    return 'Unknown rear status.'


def get_rear_object_detection_output(status):
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

    return {
        'warning': warning,
        'danger': danger,
    }


def get_rear_stop_conditions(status, lanes_xy, status_duration, lane_missing_duration, foot_monitor_armed):
    stop_conditions = []
    if foot_monitor_armed:
        if status == 'Left out' and status_duration >= FOOT_OUT_STOP_SECONDS:
            stop_conditions.append('Left foot out')
        if status == 'Right out' and status_duration >= FOOT_OUT_STOP_SECONDS:
            stop_conditions.append('Right foot out')
        if status == 'Both out' and status_duration >= FOOT_OUT_STOP_SECONDS:
            stop_conditions.append('Both feet out')
        if status == 'No feet detected' and status_duration >= NO_FEET_STOP_SECONDS:
            stop_conditions.append('No feet detected')
    if lane_missing_duration >= REAR_OUT_OF_LANE_STOP_SECONDS:
        stop_conditions.append('No lane Detected')
        stop_conditions.append('Robot out of lane')
    return stop_conditions


def open_source(source):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open source: {source}')
    return cap

#TODO: added two helpers for angle calculation
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


def main():
    args = parse_args()
    cfg = Config.fromfile(resolve_path(args.config))

    if args.conf is not None:
        cfg.test_parameters.conf_threshold = args.conf
    if args.max_lanes is not None:
        cfg.max_lanes = args.max_lanes
        cfg.test_parameters.nms_topk = args.max_lanes

    device = choose_device(args.device)
    print(f'Using device: {device}')

    model = build_net(cfg)
    load_checkpoint(model, resolve_path(args.checkpoint), device)
    model.to(device)
    model.eval()

    pose_model = None
    if not args.no_pose:
        try:
            YOLO = importlib.import_module('ultralytics').YOLO
        except Exception as e:
            raise RuntimeError('Ultralytics is required for pose overlay. Install with: pip install ultralytics') from e
        pose_path = resolve_path(args.pose_model)
        pose_model = YOLO(pose_path)
        print(f'Loaded pose model: {pose_path}')

    cap = open_source(args.source)

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    display_w, display_h = cfg.ori_img_w, cfg.ori_img_h
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (display_w, display_h))

    last_status = None
    frame_idx = 0
    log_interval = max(0.0, float(args.log_interval))
    last_log_ts = time.time()
    status_since_ts = None
    lane_missing_since_ts = None
    foot_monitor_armed = False
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        frame_idx += 1

        vis_frame, tensor = preprocess_frame(frame, cfg, device)
        with torch.inference_mode():
            output = model({'img': tensor})
            lanes = model.heads.get_lanes(output)[0]

        lanes_xy = extract_lane_xy(lanes, cfg, vis_frame.shape)
        draw_lanes(vis_frame, lanes, cfg, line_width=args.line_width)

        lane_missing_now = (lanes_xy is None or len(lanes_xy) == 0)
        if lane_missing_now:
            if lane_missing_since_ts is None:
                lane_missing_since_ts = now
        else:
            lane_missing_since_ts = None

        if pose_model is not None:
            left_ankle, right_ankle = pD.get_ankle(vis_frame, pose_model)
            status, left_in, right_in = feet_status(left_ankle, right_ankle, lanes_xy)

            safe = status == 'Safe'
            status_color = (0, 255, 0) if safe else (0, 0, 255)

            left_color = (0, 255, 0) if left_in else (0, 0, 255)
            right_color = (0, 255, 0) if right_in else (0, 0, 255)
            draw_foot(vis_frame, normalize_point(left_ankle), left_color, 'L')
            draw_foot(vis_frame, normalize_point(right_ankle), right_color, 'R')

            #TODO: added midpoint visualization and angle calculation

            # Get the midpoint between the left and right ankles and draw it
            midpoint = calculate_midpoint(left_ankle, right_ankle)
            draw_foot(vis_frame, normalize_point(midpoint), left_color, 'M')
            # Get the angle for lidar
            angle = calculate_angle_to_center(midpoint, vis_frame)
            visualize_angle(vis_frame, angle)

            cv2.putText(vis_frame,
                        status,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        status_color,
                        2,
                        cv2.LINE_AA)

            detection_output = get_rear_object_detection_output(status)

            if status != last_status:
                status_since_ts = now
            else:
                if status_since_ts is None:
                    status_since_ts = now

            status_duration = 0.0 if status_since_ts is None else (now - status_since_ts)
            lane_missing_duration = 0.0 if lane_missing_since_ts is None else (now - lane_missing_since_ts)
            if status == 'Safe':
                foot_monitor_armed = True
            stop_conditions = get_rear_stop_conditions(
                status,
                lanes_xy,
                status_duration,
                lane_missing_duration,
                foot_monitor_armed,
            )

            last_status = status

            if log_interval > 0 and (now - last_log_ts) >= log_interval:
                print(f'Object detection: {detection_output}')
                print(f'Stopping the Robot: {stop_conditions}')
                last_log_ts = now

        fps_now = 1.0 / max(1e-6, now - prev)
        prev = now
        cv2.putText(vis_frame,
                    f'FPS: {fps_now:.1f} | q: quit',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

        cv2.imshow('CLRNet Live Demo', vis_frame)
        if writer is not None:
            writer.write(vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
