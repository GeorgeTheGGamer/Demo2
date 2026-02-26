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

from clrnet.models.registry import build_net
from clrnet.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='CLRNet live demo (webcam/video)')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file (.pth)')
    parser.add_argument('--source', default='0',
                        help='camera index (e.g. 0) or video file path')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='inference device')
    parser.add_argument('--conf', type=float, default=None,
                        help='override confidence threshold from config')
    parser.add_argument('--max-lanes', type=int, default=None,
                        help='override maximum lanes shown')
    parser.add_argument('--line-width', type=int, default=4)
    parser.add_argument('--output', default=None,
                        help='optional output video path (e.g. demo.mp4)')
    return parser.parse_args()


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


def open_source(source):
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

    cap = open_source(args.source)

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

        draw_lanes(vis_frame, lanes, cfg, line_width=args.line_width)

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
