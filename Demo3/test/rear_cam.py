"""Rear camera local test using laptop webcam.
Mimics the rear-camera path in laptop_server_new.py without Pi streams.
Press 'q' to quit.
"""

import importlib
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
from Demo3.vision.frame_processor import preprocess_frame, draw_ankle_point, visualize_rear, draw_lanes
from Demo3.vision.helpers.YOLO_helpers import *
from Demo3.vision.helpers.focus_helper import FocusHelper
from Demo3.vision.helpers.lane_fixer import LaneFixer
from Demo3.vision.helpers.lane_helpers import *
from Demo3.vision.helpers.steering_helper import calculate_angle_to_center, get_rear_servo
from Demo3.connection.belt import run_esp32_status_server, push_rear_status_to_esp32
from Demo3.connection.commands import run_tcp_server, send_angles_to_pi, forward_command_to_pi

# TODO: check payloads sent to Pi and ESP32 for correctness during testing
def main():
    print("[Test] Starting TCP server thread...")
    # Start TCP server in a separate thread (non-blocking).
    threading.Thread(target=run_tcp_server, daemon=True).start()

    print("[TEST] Starting ESP32 status server thread...")
    # Start ESP32 status server in a separate thread (non-blocking).
    threading.Thread(target=run_esp32_status_server, daemon=True).start()

    print("[TEST] Loading rear CV models...")

    device = choose_device(DEFAULT_DEVICE)

    cfg = Config.fromfile(resolve_path(DEFAULT_CONFIG))
    lane_model = build_net(cfg)
    load_checkpoint(lane_model, resolve_path(DEFAULT_CHECKPOINT), device)
    lane_model.to(device)
    lane_model.eval()

    YOLO = importlib.import_module("ultralytics").YOLO
    rear_pose = YOLO(resolve_path(DEFAULT_REAR_POSE))

    ankle_focus = FocusHelper()
    rear_fixer = LaneFixer()

    # Optional warmup
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

    print("[TEST] Rear webcam test started. Press 'q' to quit.")
    forward_command_to_pi("START")  # Send START command to Pi at the beginning of the test
    g.is_running = True  # Set running state to True for testing purposes
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        rear_stop_conditions = []
        vis_rear, t_rear = preprocess_frame(frame, cfg, device)

        # 1) Lane detection
        with torch.inference_mode():
            out_rear = lane_model({"img": t_rear})
            lanes_rear = lane_model.heads.get_lanes(out_rear)[0]

        # 2) Lane fixing + draw
        lanes_xy_rear = extract_lane_xy(lanes_rear, cfg, vis_rear.shape)
        lanes_xy_rear = rear_fixer.fix(lanes_xy_rear, frame_width=vis_rear.shape[1])
        draw_lanes(vis_rear, lanes_xy_rear)

        # 3) Ankle / pose detection
        ankles = get_ankles(vis_rear.copy(), rear_pose)
        ankle_focus.update_frame_size(vis_rear.shape[1], vis_rear.shape[0])
        left_ankle, right_ankle = ankle_focus.focus(ankles)
        draw_ankle_point(vis_rear, normalize_point(left_ankle),  (0, 255, 255), "L ankle")
        draw_ankle_point(vis_rear, normalize_point(right_ankle), (255, 0, 255), "R ankle")

        # 4) Rear angle + visualization
        # Normalize first so undetected (0,0) YOLO tensors become None
        _l_norm = normalize_point(left_ankle)
        _r_norm = normalize_point(right_ankle)
        midpoint = calculate_midpoint(_l_norm, _r_norm)
        draw_ankle_point(vis_rear, normalize_point(midpoint), (255, 255, 0), "M")
        angle_rear = calculate_angle_to_center(midpoint, vis_rear)
        visualize_rear(vis_rear, angle_rear)

        feet_detected = midpoint is not None
        rear_servo_val = get_rear_servo(angle_rear, feet_detected)
        send_angles_to_pi(0, rear_servo_val)  # Front servo fixed at 0 for this test

        # 5) Rear detection output + stop conditions
        rear_status, _, _ = feet_status(left_ankle, right_ankle, lanes_xy_rear)
        rear_detection_output = build_rear_detection(rear_status)

        if rear_status == "Left out":
            rear_stop_conditions.append("Left foot out")
        elif rear_status == "Right out":
            rear_stop_conditions.append("Right foot out")
        elif rear_status == "Both out":
            rear_stop_conditions.append("Both feet out")
        elif rear_status == "No feet detected":
            rear_stop_conditions.append("No feet detected")

        if len(lanes_xy_rear) == 0:
            rear_stop_conditions.append("No lane Detected")
            rear_stop_conditions.append("Robot out of lane")

        with g.state_lock:
            g.latest_state["rear"] = {
                "status": rear_status,
                "REAR_ANGLE": f"REAR_ANGLE={rear_servo_val}",
                "object_detection": rear_detection_output,
                "stop_conditions": rear_stop_conditions,
            }
        push_rear_status_to_esp32()

        cv2.putText(
            vis_rear,
            "MODE: REAR WEBCAM TEST",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            vis_rear,
            f"rear_status={rear_status} servo={rear_servo_val}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Rear Backup Camera (Webcam Test)", vis_rear)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    forward_command_to_pi("STOP")  # Ensure we send a STOP command to the Pi when exiting
    g.is_running = False  # Set running state to False when exiting
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
