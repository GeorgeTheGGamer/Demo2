import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch

try:
	from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:
	from tensorflow.lite.python.interpreter import Interpreter

from clrnet.models.registry import build_net
from clrnet.utils.config import Config


def select_camera():
	print('\n--- Select Camera ---')
	print('0: iPhone Camera')
	print('1: MacBook Pro Camera')
	choice = input('Enter number (default 1): ').strip()
	return 0 if choice == '0' else 1


def parse_args():
	parser = argparse.ArgumentParser(description='Front camera safety stop: CLRNet + SSD MobileNet')
	parser.add_argument('config', help='CLRNet config path')
	parser.add_argument('--checkpoint', required=True, help='CLRNet checkpoint (.pth)')
	parser.add_argument('--source', default='0', help='camera index or video path')
	parser.add_argument('--camera-menu', action='store_true', help='prompt camera selection')
	parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])

	parser.add_argument('--detector-model', default='checkpoints/ssd_mobilenet_v2.tflite')
	parser.add_argument('--label-path', default='checkpoints/labelmap.txt')
	parser.add_argument('--det-input-size', type=int, default=300)
	parser.add_argument('--det-conf', type=float, default=0.5)
	parser.add_argument('--stop-threshold', type=float, default=0.7,
						help='object is close if bbox height ratio >= this value')

	parser.add_argument('--line-width', type=int, default=4)
	parser.add_argument('--conf', type=float, default=None)
	parser.add_argument('--max-lanes', type=int, default=None)
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
	colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
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


def load_detector_interpreter(model_path):
	interpreter = Interpreter(model_path=model_path)
	interpreter.allocate_tensors()
	return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


def load_labels(label_path):
	if os.path.isfile(label_path):
		with open(label_path, 'r') as f:
			labels = [line.strip() for line in f.readlines() if line.strip()]
		if labels and labels[0] == '???':
			labels = labels[1:]
		return labels
	# fallback COCO
	return [
		'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
		'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
		'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
		'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
		'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
		'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
		'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
		'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
		'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
		'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
		'hair drier', 'toothbrush'
	]


def infer_objects(frame_bgr, interpreter, input_details, output_details, input_size):
	img = cv2.resize(frame_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
	input_data = np.expand_dims(img, axis=0)
	if input_details[0]['dtype'] == np.uint8:
		input_data = input_data.astype(np.uint8)
	else:
		input_data = (np.float32(input_data) - 127.5) / 127.5

	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()

	boxes = interpreter.get_tensor(output_details[0]['index'])[0]
	classes = interpreter.get_tensor(output_details[1]['index'])[0]
	scores = interpreter.get_tensor(output_details[2]['index'])[0]
	return boxes, classes, scores


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

	print('[INFO] Loading Object Detector...')
	try:
		detector, det_input_details, det_output_details = load_detector_interpreter(args.detector_model)
	except Exception as e:
		print(f'[ERROR] Could not load model: {e}')
		sys.exit(1)

	labels = load_labels(args.label_path)
	person_ids = {i for i, name in enumerate(labels) if name.lower() == 'person'}
	if not person_ids:
		person_ids = {0, 1}

	cap = open_source(args.source, camera_menu=args.camera_menu)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	cv2.namedWindow('Demo: Safety Stop', cv2.WINDOW_NORMAL)

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

		boxes, classes, scores = infer_objects(vis_frame,
											   detector,
											   det_input_details,
											   det_output_details,
											   args.det_input_size)

		h, w = vis_frame.shape[:2]
		stop_triggered = False
		in_lane_count = 0

		for i in range(len(scores)):
			score = float(scores[i])
			if score < args.det_conf:
				continue

			class_id = int(classes[i])
			if class_id not in person_ids:
				continue

			ymin, xmin, ymax, xmax = boxes[i]
			ymin = float(np.clip(ymin, 0.0, 1.0))
			xmin = float(np.clip(xmin, 0.0, 1.0))
			ymax = float(np.clip(ymax, 0.0, 1.0))
			xmax = float(np.clip(xmax, 0.0, 1.0))
			if ymax <= ymin or xmax <= xmin:
				continue

			box_x = int(xmin * w)
			box_y = int(ymin * h)
			box_w = int((xmax - xmin) * w)
			box_h = int((ymax - ymin) * h)

			# Use center of bounding box for in-lane gating.
			obj_x = box_x + box_w // 2
			obj_y = box_y + box_h // 2

			# Ignore everything outside lane.
			if lane_polygon is None:
				continue
			if obj_y < lane_y_min or obj_y > lane_y_max:
				continue
			if not point_inside_polygon((obj_x, obj_y), lane_polygon):
				continue

			in_lane_count += 1
			height_ratio = box_h / max(1.0, float(h))

			if height_ratio >= args.stop_threshold:
				color = (0, 0, 255)
				label = f'STOP! (<1m) {score:.2f}'
				stop_triggered = True
			else:
				color = (0, 255, 0)
				label = f'Human (Safe) {score:.2f}'

			cv2.rectangle(vis_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 4)
			(tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
			label_top = max(0, box_y - 35)
			cv2.rectangle(vis_frame, (box_x, label_top), (box_x + tw + 10, label_top + 35), color, -1)
			cv2.putText(vis_frame,
						label,
						(box_x + 5, label_top + 25),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.8,
						(255, 255, 255),
						2,
						cv2.LINE_AA)

		if stop_triggered:
			status = 'CRITICAL PROXIMITY'
			status_color = (0, 0, 255)
		elif in_lane_count > 0:
			status = 'IN-LANE OBJECT: SAFE'
			status_color = (0, 255, 0)
		else:
			status = 'NO IN-LANE OBJECT'
			status_color = (0, 255, 255)

		now = time.time()
		fps_now = 1.0 / max(1e-6, now - prev)
		prev = now

		cv2.putText(vis_frame, f'FPS: {fps_now:.1f} | q: quit', (20, 40),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.putText(vis_frame, f'Status: {status}', (20, 80),
					cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)

		if stop_triggered:
			cv2.putText(vis_frame,
						'CRITICAL PROXIMITY IN LANE',
						(50, 120),
						cv2.FONT_HERSHEY_SIMPLEX,
						1.2,
						(0, 0, 255),
						4,
						cv2.LINE_AA)

		cv2.imshow('Demo: Safety Stop', vis_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
