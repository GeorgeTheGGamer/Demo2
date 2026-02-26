import argparse
import time
import os
import sys

from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch

from clrnet.models.registry import build_net
from clrnet.utils.config import Config


class Rear_Camera:

    def __init__(self, config_path, checkpoint_path, source='0', device='auto', conf=None, max_lanes=None, line_width=4, output=None, yolo_model_path="./checkpoints/yolov8n_int8.tflite"):
        self.cfg = Config.fromfile(config_path)

        if conf is not None:
            self.cfg.test_parameters.conf_threshold = conf
        if max_lanes is not None:
            self.cfg.max_lanes = max_lanes
            self.cfg.test_parameters.nms_topk = max_lanes

        self.device = self.choose_device(device)
        print(f'Using device: {self.device}')

        self.model = build_net(self.cfg)
        self.load_checkpoint(self.model, checkpoint_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        self.cap = self.open_source(source)

        self.writer = None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        display_w, display_h = self.cfg.ori_img_w, self.cfg.ori_img_h
        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output, fourcc, fps, (display_w, display_h))

        self.yolo_model_path = yolo_model_path
        self.model_pose = YOLO(self.yolo_model_path)

    def parse_args(self):
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
        parser.add_argument('--yolo-model-path', default="./checkpoints/yolov8n_int8.tflite",help='path to YOLO model for pose estimation model')
        return parser.parse_args()


    def choose_device(self,device_flag):
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


    def load_checkpoint(self, model, checkpoint_path, device):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
        cleaned = {}
        for k, v in state.items():
            if k.startswith('module.'):
                cleaned[k[len('module.'):]] = v
            else:
                cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)


    def preprocess_frame(self, frame, cfg, device):
        frame = cv2.resize(frame, (cfg.ori_img_w, cfg.ori_img_h), interpolation=cv2.INTER_LINEAR)
        cropped = frame[cfg.cut_height:, :, :]
        resized = cv2.resize(cropped, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        return frame, tensor


    def draw_lanes(self, frame, lanes, cfg, line_width=4):
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


    def open_source(self, source):
        src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f'Unable to open source: {source}')
        return cap

    def draw_keypoint(self, img, point, label, color):
        """Draw a keypoint with a label on the image."""
        x, y = map(int, point)

        # Outer white circle (contrast)
        cv2.circle(img, (x, y), 8, (255, 255, 255), -1)

        # Inner colored circle
        cv2.circle(img, (x, y), 4, color, -1)

        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            img,
            (x + 6, y - h - 8),
            (x + 6 + w + 4, y - 2),
            (0, 0, 0),
            -1
        )

        # Label text
        cv2.putText(
            img,
            label,
            (x + 8, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def get_ankle(self, frame, model):
        """
        Return the coordinates of the ankles detected by the YOLO model.
        :param model: used YOLO model
        :param frame: input frame
        :return: left_ankle coordinate, right_ankle coordinate in [x,y] format, or None if no ankle found
        """
        left_ankle = None
        right_ankle = None

        # Run model to get boxes and keypoints
        results = model(frame, verbose=False)[0]
        boxes = results.boxes
        keypoints = results.keypoints

        # Filter person by box size, bigger box size, higher probability
        max_size = 0
        for i, box in enumerate(boxes):
            # Get box details
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            size = x2 - x1
            if conf > 0.5 and size > max_size:
                max_size = size
                # 15 for left ankle, 16 for right ankle
                left_ankle = keypoints.xy[i][15]
                right_ankle = keypoints.xy[i][16]

        return left_ankle, right_ankle

    def capture(self):
        """Capture video frames, run lane detection and pose estimation, and display results."""
        prev = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            # preprocess frame for lane detection
            vis_frame, tensor = self.preprocess_frame(frame, self.cfg, self.device)

            # get ankle keypoints
            left_ankle, right_ankle = self.get_ankle(frame, self.model_pose)
            if left_ankle is not None:
                self.draw_keypoint(vis_frame, left_ankle, 'L Ankle', (0, 255, 0))
            if right_ankle is not None:
                self.draw_keypoint(vis_frame, right_ankle, 'R Ankle', (0, 255, 0))

            # run lane detection
            with torch.no_grad():
                output = self.model({'img': tensor})
                lanes = self.model.heads.get_lanes(output)[0]

            self.draw_lanes(vis_frame, lanes, self.cfg, line_width=4)

            now = time.time()
            fps_now = 1.0 / max(1e-6, now - prev)
            prev = now

            # Display FPS and quit instruction
            cv2.putText(vis_frame,
                        f'FPS: {fps_now:.1f} | q: quit',
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

            cv2.imshow('CLRNet Live Demo', vis_frame)

            if self.writer is not None:
                self.writer.write(vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()


def main():

    rear = Rear_Camera(
        config_path=os.path.join(PROJECT_ROOT, "configs/clrnet/clr_resnet18_tusimple.py"),
        checkpoint_path=os.path.join(PROJECT_ROOT, "checkpoints/tusimple_r18.pth"),
        source="0",
        device="auto",
        conf=0.5,
        max_lanes=4,
        line_width=4,
        output=None
    )

    rear.capture()

if __name__ == '__main__':
    main()
