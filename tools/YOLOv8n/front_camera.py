import argparse
import math
import time
import os
import sys
from enum import Enum

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import importlib

from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from tools.helper.steering_helper import SteeringHelper
from tools.helper.hough_lane_detect import LaneDetector
import tools.YOLOv8n.objectDetector as oD

# -----------------------------
# Local run defaults (edit here)
# -----------------------------
DEFAULT_CONFIG = 'configs/clrnet/clr_resnet18_tusimple.py'
DEFAULT_CHECKPOINT = 'checkpoints/tusimple_r18.pth'
DEFAULT_SOURCE = '0'
DEFAULT_DEVICE = 'auto'  # 'auto' | 'cuda' | 'mps' | 'cpu'
DEFAULT_YOLO_MODEL = 'checkpoints/yolov8n_int8.tflite'
CLOSE_RATIO = 0.7  # Threshold for classifying objects as 'close' based on bounding box area ratio to frame area
ACTIVE_HOUGH = False  # Whether to use Hough-based lane detection for steering angle estimation, which is more reliable on curves in real-world testing
VISUALIZATION = True # Whether to active visualization
BIG_ANGLE_THRESHOLD = 70 # Threshold in degrees for classifying steering angle as unusually large, which may indicate the robot is in a difficult situation (e.g. sharp turn, off-road, etc.) and trigger a warning message
OBJECT_IN_LANE_STOP_SECONDS = 1.0 # Continuous object-in-lane duration before stop condition
PROLONGED_DANGER_SECONDS = 5.0 # Continuous object-in-lane duration before prolonged-time stop condition
OUT_OF_LANE_STOP_SECONDS = 2.0 # Continuous out-of-lane/no-lane duration before stop condition
CORNER_EXTREME_STOP_SECONDS = 3.0 # Continuous extreme-angle duration before stop condition
LOG_INTERVAL_SECONDS = 5.0 # Periodic console output interval

LANE_COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        ]

class RobotStatus(Enum):
    NORMAL = 0
    OUT_OF_LANE = 1
    LARGE_ANGLE = 2

class AlertMessage:
    def __init__(self, obj_type = 'none', level = 'safe', side = 'none'):
        self.obj_type = obj_type # 'person' or 'obstacle' or 'none'
        self.level = level  # 'warning' or 'danger' or 'safe'
        self.side = side # 'left' or 'right' or 'none'

class FrontCamera:
    """
    CLRNet live demo for front camera lane detection and object classification with YOLO overlay.
    Usage: python tools/front_camera.py [config] [--checkpoint CHECKPOINT_PATH] [--source CAMERA_ID]
    [--device DEVICE_TYPE] [--conf LANE_CONF] [--max-lanes MAX_LANES] [--line-width LINE_WIDTH]
    [--obj-conf OBJ_CONF] [--no-objects NO_OBJECT_DETECTION] [--yolo-model YOLO_MODEL_PATH]
    [--output VIDEO_OUTPUT_PATH] [--visualization] [--close-ratio CLOSE_RATIO] [--active-hough]
    run_rear() is the main function for demo execution
    process() is the function that processes each frame, including preprocessing, lane detection, object classification, and visualization.
    """

    @staticmethod
    def parse_args():
        """Parse command line arguments for the CLRNet live demo."""
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
        parser.add_argument('--visualization', action='store_true', default=VISUALIZATION)
        parser.add_argument('--close-ratio', type=float, default=CLOSE_RATIO)
        parser.add_argument('--active-hough', action='store_true', default=ACTIVE_HOUGH)
        parser.add_argument('--yolo-stride', type=int, default=2,
                    help='run YOLO every N frames (reuse previous detections between runs)')
        parser.add_argument('--log-interval', type=float, default=LOG_INTERVAL_SECONDS,
                help='print status every N seconds (0 disables periodic status logs)')

        return parser.parse_args()

    def __init__(self):
        # Get initial configuration and model
        self.args = self.parse_args()

        # Load CLRNet configuration
        self.cfg = Config.fromfile(self.resolve_path(self.args.config))
        if self.args.conf is not None:
            self.cfg.test_parameters.conf_threshold = self.args.conf
        if self.args.max_lanes is not None:
            self.cfg.max_lanes = self.args.max_lanes
            self.cfg.test_parameters.nms_topk = self.args.max_lanes

        # Choose device for inference
        self.device = self.choose_device()
        print(f'Using device: {self.device}')

        # Build CLRNet model and load checkpoint
        self.model = build_net(self.cfg)
        self.load_checkpoint(self.resolve_path(self.args.checkpoint))
        self.model.to(self.device)
        self.model.eval()

        # Load YOLO model for object detection if not disabled
        self.object_model = None
        self.object_names = None
        if not self.args.no_objects:
            try:
                YOLO = importlib.import_module('ultralytics').YOLO
            except Exception as e:
                raise RuntimeError(
                    'Ultralytics is required for object overlay. Install with: pip install ultralytics') from e
            yolo_path = self.resolve_path(self.args.yolo_model)
            self.object_model = YOLO(yolo_path)
            self.object_names = self.object_model.names if hasattr(self.object_model, 'names') else None
            print(f'Loaded YOLO model: {yolo_path}')

        # Initialize variables for processing
        self.tensor = None # Preprocessed tensor ready for model input
        self.lanes_xy = None # Extracted lane points in pixel coordinates for visualization and object classification

        # For object detection and classification
        self.objects = None # Detected object bounding boxes
        self.danger_names = [] # Objects classified as 'danger' with their names and sides
        self.warning_names = [] # Objects classified as 'warning' with their names and sides
        self.alert_objects = []  # List of objects with added alert fields for visualization
        self.close_ratio = self.args.close_ratio # Threshold for classifying objects as 'close' based on bounding box area ratio to frame area

        # Hough-based lane detector for steering angle (more reliable on curves in real-world testing)
        if self.args.active_hough:
            self.hough_detector = LaneDetector(history_size=5)

        # For OpenCV video processing
        self.writer = None # Video writer for output if enabled
        self.cap = None # Video capture object for reading frames from source
        self.frame = None  # Current frame to be processed and visualized
        self.source = self.args.source # Source for video input (camera index or video file path)
        self.line_width = self.args.line_width # Line width for lane visualization
        self.visualization = self.args.visualization # Whether to visualize
        self.prev = time.time() # For calculating FPS
        self.fps = 0.0 # Smoothed FPS value for on-screen display
        self.last_alert_output = None # To avoid printing duplicate alerts in the console
        self.frame_idx = 0 # Current frame index for stride-based scheduling
        self.cached_objects = [] # Cached YOLO detections reused between stride steps
        self.yolo_stride = max(1, int(self.args.yolo_stride))
        self.log_interval = max(0.0, float(self.args.log_interval))
        self.last_log_ts = time.time()
        self.danger_since_ts = None # Start time for continuous object-in-lane danger
        self.out_of_lane_since_ts = None # Start time for continuous out-of-lane/no-lane condition
        self.corner_angle_extreme = False # True when raw steering angle exceeds BIG_ANGLE_THRESHOLD
        self.extreme_angle_deg = None # Angle value when corner angle is too extreme
        self.extreme_corner_since_ts = None # Start time for continuous extreme-angle condition

        # For output
        self.steer_angle = None # (Radius) Calculated steering angle based on lane detection, can be used for visualization or further processing
        self.alert_message = AlertMessage()
        self.robot_stat = RobotStatus.NORMAL

    """-----------------------------------------------------------------------------------------------------"""
    """Methods for initialization and setup, including path resolution, device selection, checkpoint loading"""
    """-----------------------------------------------------------------------------------------------------"""

    @staticmethod
    def resolve_path(path):
        """Resolve a file path that can be either absolute or relative to the project root."""
        if os.path.isabs(path):
            return path
        cwd_candidate = os.path.abspath(path)
        if os.path.exists(cwd_candidate):
            return cwd_candidate
        root_candidate = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(root_candidate):
            return root_candidate
        return cwd_candidate

    def choose_device(self):
        """Choose the appropriate torch device based on the device_flag or availability if device_flag is auto."""
        device_flag = self.args.device
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

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from a checkpoint file, handling potential 'module.' prefixes."""

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
        cleaned = {}
        for k, v in state.items():
            if k.startswith('module.'):
                cleaned[k[len('module.'):]] = v
            else:
                cleaned[k] = v
        self.model.load_state_dict(cleaned, strict=False)

    """-----------------------------------------------"""
    """Methods for video processing and lane detection"""
    """-----------------------------------------------"""

    def preprocess_frame(self, frame):
        """Preprocess the input frame for lane detection. Resizes and normalizes the image,
        and converts it to a tensor."""
        cfg = self.cfg
        self.frame = cv2.resize(frame, (cfg.ori_img_w, cfg.ori_img_h), interpolation=cv2.INTER_LINEAR)
        cropped = self.frame[cfg.cut_height:, :, :]
        resized = cv2.resize(cropped, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        self.tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

    @staticmethod
    def draw_lanes(lanes, frame, line_width):
        """Draw detected lane lines on the frame using the extracted lane points in pixel coordinates."""
        colors = LANE_COLORS

        for i, xy in enumerate(lanes):
            color = colors[i % len(colors)]
            for j in range(1, len(xy)):
                cv2.line(frame, xy[j - 1], xy[j], color, thickness=line_width)

    def extract_lane_xy(self, lanes):
        """Extract lane points in pixel coordinates, filter out points outside the frame, and sort lanes by their starting x coordinate."""
        lanes_xy = []
        frame_shape = self.frame.shape
        h, w = frame_shape[:2]
        for lane in lanes:
            pts = lane.to_array(self.cfg)
            xy = []
            for p in pts:
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < w and 0 <= y < h:
                    xy.append((x, y))
            if len(xy) >= 2:
                lanes_xy.append(xy)
        lanes_xy.sort(key=lambda xys: xys[0][0])

        self.lanes_xy = lanes_xy

    @staticmethod
    def check_out_of_lane(lanes_xy):
        """Check if the robot is out of lane if lane cannot be detected"""
        if lanes_xy is None or len(lanes_xy) == 0:
            return True
        return False

    """-------------------------------------------------"""
    """For object classification based on lane positions"""
    """-------------------------------------------------"""

    @staticmethod
    def interpolate_x_at_y(lane_xy, y):
        """Given a polyline (list of (x, y) points) and a y coordinate, interpolate the x coordinate at that y if it falls within the y range of the polyline segments."""
        for i in range(1, len(lane_xy)):
            x1, y1 = lane_xy[i - 1]
            x2, y2 = lane_xy[i]
            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
            if y_min <= y <= y_max and y1 != y2:
                t = (y - y1) / (y2 - y1)
                return x1 + t * (x2 - x1)
        return None

    def is_within_lane(self, obj):
        """Check if the box is within the lane"""
        lanes_xy = self.lanes_xy

        x1, y1, x2, y2 = obj['bbox']
        bottom_left = (x1, y2)
        bottom_mid = ((x1 + x2) / 2.0, y2)
        bottom_right = (x2, y2)
        pts = [bottom_left, bottom_mid, bottom_right]

        for pt in pts:
            x, y = pt
            xs_at_y = []
            for lane_xy in lanes_xy:
                x_interp = self.interpolate_x_at_y(lane_xy, y)
                if x_interp is not None:
                    xs_at_y.append(x_interp)

            if len(xs_at_y) < 2:
                continue

            xs_at_y.sort()
            for i in range(0, len(xs_at_y) - 1):
                if xs_at_y[i] <= x <= xs_at_y[i + 1]:
                    return True

        return False

    def get_object_name(self, obj):
        cls_id = obj['cls']
        if self.object_names is not None and cls_id in self.object_names:
            return self.object_names[cls_id]
        return str(cls_id)

    def get_side(self, obj, lane_center=None):
        """Return 'left' or 'right' based on the horizontal center of the bounding box."""
        frame_shape = self.frame.shape
        _, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = obj['bbox']
        cx = (x1 + x2) / 2.0
        return 'left' if cx < frame_w / 2.0 else 'right'

    def is_close_to(self, frame_area, obj):
        """Return True if the bounding box area ratio to the frame area exceeds the close_ratio threshold."""
        x1, y1, x2, y2 = obj['bbox']
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area_ratio = (box_w * box_h) / frame_area
        return area_ratio >= self.close_ratio

    def check_obj_stat(self, alert_obj, in_lane, is_close, side):
        """Classify object alert level and append to alert list when needed."""
        if in_lane:
            alert_obj['alert_level'] = 'danger'
            self.alert_message.level = 'danger'
            alert_obj['alert_text'] = f"Danger {alert_obj['name']} in lane"
            self.alert_message.obj_type = 'person' if alert_obj['name'] == 'person' else 'obstacles'
            alert_obj['alert_color'] = (0, 0, 255)
            alert_obj['side'] = side
            self.alert_message.side = side
            self.alert_objects.append(alert_obj)
        elif is_close:
            alert_obj['alert_level'] = 'warning'
            self.alert_message.level = 'warning'
            alert_obj['alert_text'] = f"Warning {alert_obj['name']} close"
            self.alert_message.obj_type = 'person' if alert_obj['name'] == 'person' else 'obstacles'
            alert_obj['alert_color'] = (0, 165, 255)
            alert_obj['side'] = side
            self.alert_message.side = side
            self.alert_objects.append(alert_obj)
        else:
            self.alert_message.level = 'safe'
            self.alert_message.obj_type = 'none'
            self.alert_message.side = 'none'

    def classify_objects(self):
        """Classify detected objects as 'danger' if they are within lane boundaries,
        or 'warning' if they are close to the lane (based on bounding box area ratio
        compared to frame area). Returns a list of alert objects with added fields:
        Also returns sorted lists of unique (name, side) tuples for warnings and dangers.
        """
        frame_shape = self.frame.shape
        frame_h, frame_w = frame_shape[:2]
        frame_area = max(1.0, float(frame_w * frame_h))

        # Check objects status (close or in lane)
        for obj in self.objects:
            in_lane = self.is_within_lane(obj)
            is_close = self.is_close_to(frame_area, obj)
            side = self.get_side(obj)
            alert_obj = dict(obj)
            alert_obj['name'] = self.get_object_name(obj)  # resolve name once, reuse everywhere
            self.check_obj_stat(alert_obj, in_lane, is_close, side)

    def draw_objects(self):
        """Draw bounding boxes and labels for detected objects on the frame.
        Uses alert text and color if available, otherwise defaults to class name and confidence."""
        for obj in self.alert_objects:
            # Get box details
            x1, y1, x2, y2 = [int(round(v)) for v in obj['bbox']]
            conf = obj['conf']
            label_name = obj.get('name', str(obj['cls']))
            label = f'{label_name} {conf:.2f}'

            # Set color based on alert level if available, otherwise default to orange
            color = obj.get('alert_color', (0, 165, 255))

            # Prepend alert text to label if available
            alert_text = obj.get('alert_text')
            if alert_text:
                label = f'{alert_text} | {label}'

            # Visualize bounding box and label on the frame
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA)

    def draw_fps(self):
        """Draw FPS counter on the top-left corner of the frame."""
        label = f'FPS: {self.fps:.1f}'
        cv2.putText(self.frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

    def get_outline(self):
        """
        Generate alert outline strings for display based on the classified alert objects.
        Returns a list of strings summarizing the warnings and dangers, including object classes and sides.
        """
        danger_objs = [obj for obj in self.alert_objects if obj.get('alert_level') == 'danger']
        warning_objs = [obj for obj in self.alert_objects if obj.get('alert_level') == 'warning']

        out_lines = []
        if warning_objs:
            warning_part = ', '.join([f"{obj['name']}({obj['side']})" for obj in warning_objs])
            out_lines.append(f"Warning: {warning_part} close")
        if danger_objs:
            danger_part = ', '.join([f"{obj['name']}({obj['side']})" for obj in danger_objs])
            out_lines.append(f"Danger: {danger_part} in lane")
        return out_lines

    def get_object_detection_output(self):
        """Build a compact dictionary-style object detection summary for console output."""
        warning_objs = [obj for obj in self.alert_objects if obj.get('alert_level') == 'warning']
        danger_objs = [obj for obj in self.alert_objects if obj.get('alert_level') == 'danger']

        def summarize(objs):
            counts = {}
            for obj in objs:
                key = (obj.get('name', 'unknown'), obj.get('side', 'none'))
                counts[key] = counts.get(key, 0) + 1

            parts = []
            for (name, side), count in counts.items():
                label = f'{name}({side})'
                if count > 1:
                    label = f'{label}x{count}'
                parts.append(label)
            return parts

        warning_summary = summarize(warning_objs)
        danger_summary = summarize(danger_objs)

        return {
            'warning': warning_summary,
            'danger': danger_summary,
        }

    def get_stop_conditions(self, now_ts):
        """Return stop-condition reasons for front camera logic."""
        stop_conditions = []

        if self.danger_since_ts is not None and (now_ts - self.danger_since_ts) >= OBJECT_IN_LANE_STOP_SECONDS:
            stop_conditions.append('If object is in lane')
        if self.danger_since_ts is not None and (now_ts - self.danger_since_ts) >= PROLONGED_DANGER_SECONDS:
            stop_conditions.append('For prolonged time')
        if self.out_of_lane_since_ts is not None and (now_ts - self.out_of_lane_since_ts) >= OUT_OF_LANE_STOP_SECONDS:
            stop_conditions.append('No lane Detected')
            stop_conditions.append('Robot out of lane')
        if self.extreme_corner_since_ts is not None and (now_ts - self.extreme_corner_since_ts) >= CORNER_EXTREME_STOP_SECONDS:
            stop_conditions.append('Corner Angle too extreme')

        return stop_conditions

    def update_front_condition_timers(self, now_ts, out_of_lane_now, angle_extreme_now, has_object_danger):
        """Update continuous-condition timers for front stop logic."""
        if out_of_lane_now:
            if self.out_of_lane_since_ts is None:
                self.out_of_lane_since_ts = now_ts
        else:
            self.out_of_lane_since_ts = None

        if angle_extreme_now:
            if self.extreme_corner_since_ts is None:
                self.extreme_corner_since_ts = now_ts
        else:
            self.extreme_corner_since_ts = None

        if has_object_danger:
            if self.danger_since_ts is None:
                self.danger_since_ts = now_ts
        else:
            self.danger_since_ts = None

    """-----------"""
    """For Open CV"""
    """-----------"""

    def open_source(self):
        """Open camera"""
        source = self.args.source
        src = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f'Unable to open source: {source}')


    def ini_writer(self):
        """Init video writer"""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        display_w, display_h = self.cfg.ori_img_w, self.cfg.ori_img_h
        if self.args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.args.output, fourcc, fps, (display_w, display_h))

    """-----------------"""
    """Main Function!!!"""
    """-----------------"""

    def process(self, frame):
        """Process a single video frame: perform lane detection, object classification, and visualization."""
        now = time.time()
        dt = max(1e-6, now - self.prev)
        instant_fps = 1.0 / dt
        self.fps = instant_fps if self.fps <= 0.0 else (0.9 * self.fps + 0.1 * instant_fps)
        self.prev = now

        self.frame_idx += 1
        # 1. Preprocess frame and store the frame and preprocessed tensor for model input
        self.preprocess_frame(frame)
        # A copy of the original frame before drawing lanes and objects,
        # used for object detection to avoid interference from lane drawings
        detect_frame = self.frame.copy()


        # 2. Extract lane polylines and draw on frame
        with torch.inference_mode():
            output = self.model({'img': self.tensor})
            lanes = self.model.heads.get_lanes(output)[0]
        self.extract_lane_xy(lanes)

        # Check if robot is out of lane based on lane points, update robot status accordingly
        out_of_lane_now = self.check_out_of_lane(self.lanes_xy)
        self.robot_stat = RobotStatus.OUT_OF_LANE if out_of_lane_now else RobotStatus.NORMAL

        # 3. Calculate and store steering angle
        # Use hough-based lane detection for steering angle estimation if active
        hough_lanes = None
        frame_shape = self.frame.shape
        frame_h, frame_w = frame_shape[:2]
        if self.args.active_hough:
            hough_lanes = self.hough_detector.detect(detect_frame)
            steer_helper = SteeringHelper(hough_lanes, frame_width=frame_w, n_samples=20, threshold=10)
        # Use CLRNet lane points for steering angle calculation if Hough is not active
        else:
            steer_helper = SteeringHelper(self.lanes_xy, frame_width=frame_w, n_samples=20, threshold=10)
        max_angle = math.radians(45)
        angle = steer_helper.heading_angle
        self.corner_angle_extreme = False
        self.extreme_angle_deg = None
        # Check if angle is weirdly large, if so send danger message
        angle_extreme_now = abs(angle) >= math.radians(BIG_ANGLE_THRESHOLD)
        if angle_extreme_now:
            self.robot_stat = RobotStatus.LARGE_ANGLE
            self.corner_angle_extreme = True
            self.extreme_angle_deg = math.degrees(angle)

        angle = max(min(angle, max_angle), -max_angle)
        self.steer_angle = angle

        # 4. Perform object detection and classify objects based on lane positions
        self.alert_objects = []
        self.alert_message = AlertMessage()
        if self.object_model is not None:
            if self.frame_idx % self.yolo_stride == 0:
                self.cached_objects = oD.get_objects(detect_frame, self.object_model, conf_thres=self.args.obj_conf)
            self.objects = self.cached_objects
            self.classify_objects()

            has_object_danger = any(obj.get('alert_level') == 'danger' for obj in self.alert_objects)
            self.update_front_condition_timers(now, out_of_lane_now, angle_extreme_now, has_object_danger)

            out_lines = self.get_outline()
            detection_output = self.get_object_detection_output()
            stop_conditions = self.get_stop_conditions(now)

            # 5. Visualization part: draw lanes, objects, and alert text on the frame.
            if self.visualization:
                steer_helper.visualization(self.frame)
                self.draw_objects()
                self.draw_lanes(self.lanes_xy, self.frame, self.line_width)
                if self.args.active_hough:
                    self.draw_lanes(hough_lanes, self.frame, self.line_width*2)
                for i, line in enumerate(out_lines):
                    color = (0, 165, 255) if line.startswith('Warning') else (0, 0, 255)
                    cv2.putText(self.frame,
                                line,
                                (20, 80 + i * 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                color,
                                2,
                                cv2.LINE_AA)
                self.draw_fps()

            # Periodic status output to reduce console I/O overhead
            if self.log_interval > 0 and (now - self.last_log_ts) >= self.log_interval:
                print(f"Robot Status: {self.robot_stat.name}, Steering Angle: {math.degrees(self.steer_angle):.2f} degrees")
                print(f"Object detection: {detection_output}")
                print(f"Stopping the Robot: {stop_conditions}")
                self.last_log_ts = now
        elif self.visualization:
            # Keep FPS visible even when object detection is disabled
            self.draw_fps()

    def run_rear(self):
        """Main loop for processing video frames, performing lane detection, object classification, and visualization."""
        # Open video source
        self.open_source()
        self.ini_writer()

        # Start capturing
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            self.process(frame)

            cv2.imshow('CLRNet Live Demo', self.frame)
            if self.writer is not None:
                self.writer.write(self.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()

def main():
    demo = FrontCamera()
    demo.run_rear()

if __name__ == '__main__':
    main()
