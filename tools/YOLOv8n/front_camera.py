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
import objectDetector as oD


# -----------------------------
# Local run defaults (edit here)
# -----------------------------
DEFAULT_CONFIG = 'configs/clrnet/clr_resnet18_tusimple.py'
DEFAULT_CHECKPOINT = 'checkpoints/tusimple_r18.pth'
DEFAULT_SOURCE = '0'
DEFAULT_DEVICE = 'mps'  # 'auto' | 'cuda' | 'mps' | 'cpu'
DEFAULT_YOLO_MODEL = 'checkpoints/yolov8n_int8.tflite'


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
    parser.add_argument('--obj-conf', type=float, default=0.3,
                        help='object confidence threshold for YOLO overlay')
    parser.add_argument('--no-objects', action='store_true',
                        help='disable YOLO object overlay')
    parser.add_argument('--yolo-model', default=DEFAULT_YOLO_MODEL,
                        help='YOLO model path (.pt/.tflite) relative to project root or absolute')
    parser.add_argument('--output', default=None,
                        help='optional output video path (e.g. demo.mp4)')
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


def get_object_name(obj, names=None):
    cls_id = obj['cls']
    if names is not None and cls_id in names:
        return names[cls_id]
    return str(cls_id)


def classify_objects(objects, lanes_xy, frame_shape, names=None, close_ratio=0.7):
    alert_objects = []
    danger_names = set()
    warning_names = set()
    frame_h, frame_w = frame_shape[:2]
    frame_area = max(1.0, float(frame_w * frame_h))

    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        bottom_left = (x1, y2)
        bottom_mid = ((x1 + x2) / 2.0, y2)
        bottom_right = (x2, y2)
        left_in = is_center_within_lane(bottom_left, lanes_xy)
        mid_in = is_center_within_lane(bottom_mid, lanes_xy)
        right_in = is_center_within_lane(bottom_right, lanes_xy)
        in_lane = left_in or mid_in or right_in

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area_ratio = (box_w * box_h) / frame_area
        is_close = area_ratio >= close_ratio

        obj_name = get_object_name(obj, names)
        alert_obj = dict(obj)

        if in_lane:
            alert_obj['alert_level'] = 'danger'
            alert_obj['alert_text'] = f'Danger {obj_name} in lane'
            alert_obj['alert_color'] = (0, 0, 255)  # red
            alert_objects.append(alert_obj)
            danger_names.add(obj_name)
        elif is_close:
            alert_obj['alert_level'] = 'warning'
            alert_obj['alert_text'] = f'Warning {obj_name} close'
            alert_obj['alert_color'] = (0, 165, 255)  # orange
            alert_objects.append(alert_obj)
            warning_names.add(obj_name)

    return alert_objects, sorted(warning_names), sorted(danger_names)


def draw_objects(frame, objects, names=None):
    for obj in objects:
        x1, y1, x2, y2 = [int(round(v)) for v in obj['bbox']]
        cls_id = obj['cls']
        conf = obj['conf']
        if names is not None and cls_id in names:
            label_name = names[cls_id]
        else:
            label_name = str(cls_id)
        label = f'{label_name} {conf:.2f}'
        color = obj.get('alert_color', (0, 165, 255))
        alert_text = obj.get('alert_text')
        if alert_text:
            label = f'{alert_text} | {label}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA)


def open_source(source):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open source: {source}')
    return cap


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

    object_model = None
    object_names = None
    if not args.no_objects:
        try:
            YOLO = importlib.import_module('ultralytics').YOLO
        except Exception as e:
            raise RuntimeError('Ultralytics is required for object overlay. Install with: pip install ultralytics') from e
        yolo_path = resolve_path(args.yolo_model)
        object_model = YOLO(yolo_path)
        object_names = object_model.names if hasattr(object_model, 'names') else None
        print(f'Loaded YOLO model: {yolo_path}')

    cap = open_source(args.source)

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    display_w, display_h = cfg.ori_img_w, cfg.ori_img_h
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (display_w, display_h))

    last_alert_output = None
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis_frame, tensor = preprocess_frame(frame, cfg, device)
        with torch.no_grad():
            output = model({'img': tensor})
            lanes = model.heads.get_lanes(output)[0]

        lanes_xy = extract_lane_xy(lanes, cfg, vis_frame.shape)
        detect_frame = vis_frame.copy()
        draw_lanes(vis_frame, lanes, cfg, line_width=args.line_width)

        if object_model is not None:
            objects = oD.get_objects(detect_frame, object_model, conf_thres=args.obj_conf)
            alert_objects, warning_names, danger_names = classify_objects(
                objects, lanes_xy, vis_frame.shape, object_names, close_ratio=0.7
            )
            draw_objects(vis_frame, alert_objects, object_names)

            out_lines = []
            if warning_names:
                out_lines.append(f"Warning {', '.join(warning_names)} close")
            if danger_names:
                out_lines.append(f"Danger {', '.join(danger_names)} in lane")

            for i, line in enumerate(out_lines):
                color = (0, 165, 255) if line.startswith('Warning') else (0, 0, 255)
                cv2.putText(vis_frame,
                            line,
                            (20, 80 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2,
                            cv2.LINE_AA)

            alert_output = ' | '.join(out_lines) if out_lines else None
            if alert_output and alert_output != last_alert_output:
                print(alert_output)
            last_alert_output = alert_output

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
