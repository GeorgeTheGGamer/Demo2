import argparse
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

from clrnet.models.registry import build_net
from clrnet.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='CLRNet live demo (webcam/video)')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file (.pth)')
    parser.add_argument('--source', default='0',
                        help='camera index (e.g. 0) or video file path')
    parser.add_argument('--camera-menu', action='store_true',
                        help='prompt camera selection menu (0=iPhone, 1=MacBook)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='inference device')
    parser.add_argument('--conf', type=float, default=None,
                        help='override confidence threshold from config')
    parser.add_argument('--max-lanes', type=int, default=None,
                        help='override maximum lanes shown')
    parser.add_argument('--line-width', type=int, default=4)
    parser.add_argument('--pose-model', default='checkpoints/movenet_thunder.tflite',
                        help='MoveNet TFLite model path')
    parser.add_argument('--pose-input-size', type=int, default=256,
                        help='pose model input size (thunder=256, lightning=192)')
    parser.add_argument('--pose-conf', type=float, default=0.3,
                        help='keypoint confidence threshold for feet')
    parser.add_argument('--foot-min-y-ratio', type=float, default=0.55,
                        help='minimum vertical position ratio for valid feet (filters floating points)')
    parser.add_argument('--output', default=None,
                        help='optional output video path (e.g. demo.mp4)')
    return parser.parse_args()


def select_camera():
    print('\n--- Select Camera ---')
    print('0: iPhone Camera (Continuity)')
    print('1: MacBook Pro Camera')
    choice = input('Enter number (default 1): ').strip()
    if choice == '0':
        return 0
    return 1


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

    return lanes_xy


def interp_x_at_y(lane_xy, y_target):
    if len(lane_xy) < 2:
        return None
    pts = sorted(lane_xy, key=lambda p: p[1])
    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]
    if y_target < ys[0] or y_target > ys[-1]:
        return None
    for i in range(1, len(pts)):
        y0, y1 = ys[i - 1], ys[i]
        if y0 <= y_target <= y1 and y1 != y0:
            t = (y_target - y0) / (y1 - y0)
            return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return None


def choose_ego_lane_pair(lanes_xy, width, probe_y):
    center_x = width * 0.5
    candidates = []
    for lane in lanes_xy:
        x = interp_x_at_y(lane, probe_y)
        if x is not None:
            candidates.append((x, lane))
    if len(candidates) < 2:
        return None, None

    left = [c for c in candidates if c[0] <= center_x]
    right = [c for c in candidates if c[0] > center_x]
    if not left or not right:
        return None, None

    left_lane = max(left, key=lambda c: c[0])[1]
    right_lane = min(right, key=lambda c: c[0])[1]
    return left_lane, right_lane


def build_lane_polygon(left_lane, right_lane):
    left = sorted(left_lane, key=lambda p: p[1])
    right = sorted(right_lane, key=lambda p: p[1])
    if len(left) < 2 or len(right) < 2:
        return None

    left_ys = np.array([p[1] for p in left], dtype=np.float32)
    left_xs = np.array([p[0] for p in left], dtype=np.float32)
    right_ys = np.array([p[1] for p in right], dtype=np.float32)
    right_xs = np.array([p[0] for p in right], dtype=np.float32)

    y_min = int(max(np.min(left_ys), np.min(right_ys)))
    y_max = int(min(np.max(left_ys), np.max(right_ys)))
    if y_max - y_min < 30:
        return None

    sample_ys = np.linspace(y_min, y_max, num=70, dtype=np.float32)
    left_interp = np.interp(sample_ys, left_ys, left_xs)
    right_interp = np.interp(sample_ys, right_ys, right_xs)

    left_pts = np.stack([left_interp, sample_ys], axis=1)
    right_pts = np.stack([right_interp, sample_ys], axis=1)
    polygon = np.vstack([left_pts, right_pts[::-1]])
    return polygon.astype(np.int32), y_min, y_max


def point_inside_polygon(point_xy, polygon):
    x, y = point_xy
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def load_pose_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def infer_feet(frame_bgr, interpreter, input_details, output_details, input_size, conf_threshold):
    h, w, _ = frame_bgr.shape
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(img, axis=0)

    if input_details[0]['dtype'] == np.float32:
        input_data = input_data.astype(np.float32)
    else:
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

    feet = {}
    for label, idx in [('left', 15), ('right', 16)]:
        y, x, score = keypoints[idx]
        if score > conf_threshold:
            feet[label] = {
                'xy': (int(x * w), int(y * h)),
                'score': float(score)
            }
        else:
            feet[label] = None
    return feet


def classify_foot_safety(feet, lane_polygon, lane_y_min, lane_y_max, frame_h, foot_min_y_ratio):
    min_y_px = int(frame_h * foot_min_y_ratio)
    result = {'left': None, 'right': None}

    for side in ('left', 'right'):
        f = feet[side]
        if f is None:
            result[side] = 'undetected'
            continue
        x, y = f['xy']

        if y < min_y_px:
            result[side] = 'undetected'
            continue

        if lane_polygon is None:
            result[side] = 'undetected'
            continue

        if y < lane_y_min or y > lane_y_max:
            result[side] = 'undetected'
            continue

        result[side] = 'in' if point_inside_polygon((x, y), lane_polygon) else 'out'

    left_state = result['left']
    right_state = result['right']

    if left_state in ('in', 'out') and right_state in ('in', 'out'):
        if left_state == 'in' and right_state == 'in':
            overall = 'SAFE'
        elif left_state == 'out' and right_state == 'in':
            overall = 'LEFT FOOT OUT'
        elif left_state == 'in' and right_state == 'out':
            overall = 'RIGHT FOOT OUT'
        else:
            overall = 'BOTH OUT'
    else:
        overall = 'UNDETECTED'

    return result, overall


def open_source(source, camera_menu=False):
    if camera_menu and source.isdigit():
        src = select_camera()
    else:
        src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open source: {source}')
    return cap


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.conf is not None:
        cfg.test_parameters.conf_threshold = args.conf
    if args.max_lanes is not None:
        cfg.max_lanes = args.max_lanes
        cfg.test_parameters.nms_topk = args.max_lanes

    device = choose_device(args.device)
    print(f'Using device: {device}')

    model = build_net(cfg)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    print('[INFO] Loading Pose Estimator...')
    try:
        pose_interpreter, pose_input_details, pose_output_details = load_pose_interpreter(args.pose_model)
    except Exception as e:
        print(f'[ERROR] Could not load pose model: {e}')
        sys.exit(1)

    cap = open_source(args.source, camera_menu=args.camera_menu)

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    display_w, display_h = cfg.ori_img_w, cfg.ori_img_h
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (display_w, display_h))

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis_frame, tensor = preprocess_frame(frame, cfg, device)
        with torch.no_grad():
            output = model({'img': tensor})
            lanes = model.heads.get_lanes(output)[0]

        lanes_xy = draw_lanes(vis_frame, lanes, cfg, line_width=args.line_width)

        probe_y = vis_frame.shape[0] - 20
        left_lane, right_lane = choose_ego_lane_pair(lanes_xy, vis_frame.shape[1], probe_y)
        lane_polygon = None
        lane_y_min, lane_y_max = 0, vis_frame.shape[0] - 1
        if left_lane is not None and right_lane is not None:
            lane_poly_data = build_lane_polygon(left_lane, right_lane)
            if lane_poly_data is not None:
                lane_polygon, lane_y_min, lane_y_max = lane_poly_data
                overlay = vis_frame.copy()
                cv2.fillPoly(overlay, [lane_polygon], (30, 30, 200))
                vis_frame = cv2.addWeighted(overlay, 0.15, vis_frame, 0.85, 0)

        feet = infer_feet(vis_frame,
                          pose_interpreter,
                          pose_input_details,
                          pose_output_details,
                          args.pose_input_size,
                          args.pose_conf)

        per_foot, status = classify_foot_safety(feet,
                                                lane_polygon,
                                                lane_y_min,
                                                lane_y_max,
                                                vis_frame.shape[0],
                                                args.foot_min_y_ratio)

        for side, color, text in [('left', (0, 0, 255), 'L_FOOT'), ('right', (0, 255, 0), 'R_FOOT')]:
            foot = feet[side]
            if foot is None:
                continue
            x, y = foot['xy']
            cv2.circle(vis_frame, (x, y), 9, color, -1)
            cv2.putText(vis_frame,
                        f'{text}:{per_foot[side].upper()}',
                        (x + 12, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA)

        status_color = (0, 255, 0) if status == 'SAFE' else (0, 0, 255)
        cv2.putText(vis_frame,
                    f'Status: {status}',
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    status_color,
                    2,
                    cv2.LINE_AA)

        now = time.time()
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
