"""Front camera local test using laptop webcam.
Mimics the front-camera path in laptop_server_new.py without Pi streams.
Press 'q' to quit.
"""

import importlib
import math
import os
import sys
import threading
import time
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import Demo3.states.globals as g
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from Demo3.config.bootstrap import *
from Demo3.vision.frame_processor import preprocess_frame, draw_lanes, draw_front_objects
from Demo3.vision.helpers.YOLO_helpers import *
from Demo3.vision.helpers.lane_fixer import LaneFixer
from Demo3.vision.helpers.lane_helpers import *
from Demo3.vision.helpers.steering_helper import SteeringHelper
from Demo3.connection.commands import run_tcp_server, send_angles_to_pi, forward_command_to_pi


def main():
    print("[TEST] Starting TCP server thread...")
    # Start TCP server in a separate thread (non-blocking).
    threading.Thread(target=run_tcp_server, daemon=True).start()

    print("[TEST] Loading front CV models...")

    device = choose_device(DEFAULT_DEVICE)

    cfg = Config.fromfile(resolve_path(DEFAULT_CONFIG))
    lane_model = build_net(cfg)
    load_checkpoint(lane_model, resolve_path(DEFAULT_CHECKPOINT), device)
    lane_model.to(device)
    lane_model.eval()

    YOLO = importlib.import_module("ultralytics").YOLO
    front_yolo = YOLO(resolve_path(DEFAULT_FRONT_YOLO))

    # Optional warmup (same spirit as server warmup)
    try:
        dummy = torch.zeros(1, 3, cfg.img_h, cfg.img_w, device=device)
        with torch.inference_mode():
            out_dummy = lane_model({"img": dummy})
            _ = lane_model.heads.get_lanes(out_dummy)[0]
        print("[TEST] Warmup complete.")
    except Exception as e:
        print(f"[TEST] Warmup failed (non-fatal): {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[TEST] Cannot open webcam index 0.")
        return

    lane_fixer = LaneFixer()
    frame_idx = 0
    cached_front_objects = []

    print("[TEST] Front webcam test started. Press 'q' to quit.")
    forward_command_to_pi("START")  # Ensure Pi is in START mode for testing
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        front_stop_conditions = []
        vis_front, t_front = preprocess_frame(frame, cfg, device)

        # 1) Lane detection
        with torch.inference_mode():
            out_front = lane_model({"img": t_front})
            lanes_front = lane_model.heads.get_lanes(out_front)[0]

        # 2) Lane fix + draw
        lanes_xy_front = extract_lane_xy(lanes_front, cfg, vis_front.shape)
        lanes_xy_front = lane_fixer.fix(lanes_xy_front, frame_width=vis_front.shape[1])
        draw_lanes(vis_front, lanes_xy_front)

        # 3) Object detection (every 2 frames)
        frame_idx += 1
        if frame_idx % 2 == 0:
            cached_front_objects = get_objects(vis_front.copy(), front_yolo, conf_thres=0.3)

        draw_front_objects(
            vis_front,
            cached_front_objects,
            lanes_xy_front,
            front_yolo.names if hasattr(front_yolo, "names") else None,
        )

        # 4) Build front detection output + stop conditions
        front_detection_output = build_front_detection(
            cached_front_objects,
            lanes_xy_front,
            vis_front.shape,
            front_yolo.names if hasattr(front_yolo, "names") else None,
            close_ratio=0.7,
        )
        if len(front_detection_output.get("danger", [])) > 0:
            front_stop_conditions.append("If object is in lane")
        if len(lanes_xy_front) == 0:
            front_stop_conditions.append("No lane Detected")
            front_stop_conditions.append("Robot out of lane")

        # 5) Steering angle calculation
        steer_helper = SteeringHelper(lanes_xy_front, vis_front.shape[:2], n_samples=20)
        steer_angle = max(min(steer_helper.heading_angle, MINMAX_ANGLE), -MINMAX_ANGLE)
        latest_angle_deg = round(steer_angle, 1)

        robot_status = "OUT_OF_LANE" if len(lanes_xy_front) == 0 else "NORMAL"
        if abs(steer_helper.heading_angle) >= math.radians(70):
            robot_status = "LARGE_ANGLE"
            front_stop_conditions.append("Corner Angle too extreme")

        with g.state_lock:
            g.latest_state["front"] = {
                "robot_status": robot_status,
                "FRONT_ANGLE": f"FRONT_ANGLE={latest_angle_deg}",
                "object_detection": front_detection_output,
                "stop_conditions": front_stop_conditions,
            }

        send_angles_to_pi(latest_angle_deg, 90)

        cv2.putText(
            vis_front,
            "MODE: FRONT WEBCAM TEST",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            vis_front,
            f"status={robot_status} angle={latest_angle_deg}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Front AI Camera (Webcam Test)", vis_front)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    forward_command_to_pi("STOP")  # Ensure Pi is in STOP mode after testing
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
