import importlib
import threading
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import Demo3.states.globals as g
from Demo3.connection.commands import (
    run_tcp_server,
    receive_rear_video,
    receive_front_video,
    send_angles_to_pi,
    forward_command_to_pi,
    broadcast_status,
)
from clrnet.utils.config import Config
from clrnet.models.registry import build_net
from Demo3.config.bootstrap import *
from Demo3.vision.helpers.YOLO_helpers import *
from Demo3.vision.helpers.lane_helpers import *
from Demo3.vision.frame_processor import *
from Demo3.vision.helpers.lane_fixer import LaneFixer
from Demo3.vision.helpers.steering_helper import SteeringHelper, get_rear_servo, calculate_angle_to_center
from Demo3.vision.helpers.focus_helper import FocusHelper

def main():

    # --- 1. Initialization ---
    print('[LAPTOP] Loading AI Models... Please wait.')
    device = choose_device(DEFAULT_DEVICE)
    cfg = Config.fromfile(resolve_path(DEFAULT_CONFIG))
    lane_model = build_net(cfg)
    load_checkpoint(lane_model, resolve_path(DEFAULT_CHECKPOINT), device)
    lane_model.to(device)
    lane_model.eval()
    YOLO = importlib.import_module('ultralytics').YOLO
    front_yolo = YOLO(resolve_path(DEFAULT_FRONT_YOLO))
    rear_pose = YOLO(resolve_path(DEFAULT_REAR_POSE))
    ankle_focus = FocusHelper()
    front_fixer = LaneFixer()
    rear_fixer = LaneFixer()
    print(f'[LAPTOP] Models loaded on {device}. Ready!')

    # --- 2. Synthetic Warmup: force MPS kernel compilation before any real frame ---
    print('[CV] Running synthetic warmup to compile MPS kernels...')
    try:
        dummy = torch.zeros(1, 3, cfg.img_h, cfg.img_w, device=device)
        with torch.inference_mode():
            out_dummy = lane_model({'img': dummy})
            _ = lane_model.heads.get_lanes(out_dummy)[0]
        print('[CV] ✅ Synthetic warmup complete — MPS kernels compiled')
    except Exception as e:
        print(f'[CV] ⚠️  Synthetic warmup failed (non-fatal): {e}')

    # --- 3. Start Background Threads ---
    threading.Thread(target=run_tcp_server, daemon=True).start()
    threading.Thread(target=receive_rear_video, daemon=True).start()
    threading.Thread(target=receive_front_video, daemon=True).start()

    # --- 4. Pre-loop Setup ---
    print('[CV] Waiting for first real front frame to confirm CV readiness...')
    frame_idx = 0
    cached_front_objects = []
    front_placeholder = make_placeholder_frame('Front AI Camera')
    rear_placeholder = make_placeholder_frame('Rear Backup Camera')
    print("[SERVER] 🖥️ Displaying video feeds. Press 'q' to quit.")

    while True:
        # --- 5. Frame Acquisition ---
        now_ts = time.time()
        front = g.latest_front_frame if (g.front_frame_ts > 0 and (now_ts - g.front_frame_ts) <= FRAME_MAX_AGE_SEC) else None
        rear  = g.latest_rear_frame  if (g.rear_frame_ts  > 0 and (now_ts - g.rear_frame_ts)  <= FRAME_MAX_AGE_SEC) else None
        front_display = front_placeholder.copy() if front is None else front.copy()
        rear_display  = rear_placeholder.copy()  if rear  is None else rear.copy()

        # --- 6. CV Warmup Check: run one inference as soon as first frame arrives ---
        if not g.cv_ready and front is not None:
            try:
                _, t_warmup = preprocess_frame(front, cfg, device)
                with torch.inference_mode():
                    out_w = lane_model({'img': t_warmup})
                    _ = lane_model.heads.get_lanes(out_w)[0]
                g.cv_ready = True
                g.cv_ready_event.set()
                print('[CV] ✅ CV is ready — robot can now be started from the app')
            except Exception as e:
                print(f'[CV] ⚠️  Warmup inference failed: {e}')

        # --- 7. Condition Lists Reset ---
        front_stop_conditions = []
        rear_stop_conditions = []

        if g.is_running:
            latest_angle_deg = 0.0
            rear_servo_val = 90

            # --- 8. Front Camera Processing ---
            if front is not None:
                frame_idx += 1
                vis_front, t_front = preprocess_frame(front, cfg, device)

                # 8a. Lane detection
                with torch.inference_mode():
                    out_front = lane_model({'img': t_front})
                    lanes_front = lane_model.heads.get_lanes(out_front)[0]

                # 8b. Lane fixing & drawing
                lanes_xy_front = extract_lane_xy(lanes_front, cfg, vis_front.shape)
                lanes_xy_front = front_fixer.fix(lanes_xy_front, frame_width=vis_front.shape[1])
                draw_lanes(vis_front, lanes_xy_front)

                # 8c. Object detection (every 2 frames)
                if frame_idx % 2 == 0:
                    cached_front_objects = get_objects(vis_front.copy(), front_yolo, conf_thres=0.3)

                draw_front_objects(
                    vis_front,
                    cached_front_objects,
                    lanes_xy_front,
                    front_yolo.names if hasattr(front_yolo, 'names') else None,
                )

                # 8d. Build detection output & append stop conditions
                front_detection_output = build_front_detection(
                    cached_front_objects,
                    lanes_xy_front,
                    vis_front.shape,
                    front_yolo.names if hasattr(front_yolo, 'names') else None,
                    close_ratio=0.7,
                )
                if len(front_detection_output['danger']) > 0:
                    front_stop_conditions.append('If object is in lane')
                if len(lanes_xy_front) == 0:
                    front_stop_conditions.append('No lane Detected')
                    front_stop_conditions.append('Robot out of lane')

                # 8e. Steering angle calculation
                steer_helper = None
                if g.state == 'STRAIGHT':
                    steer_helper = SteeringHelper(lanes_xy_front, vis_front.shape[:2], n_samples=20)
                if g.state == 'LEFT' or g.state == 'RIGHT':
                    steer_helper = SteeringHelper(lanes_xy_front, vis_front.shape[:2], n_samples=20)
                steer_angle = max(min(steer_helper.heading_angle, MINMAX_ANGLE), MINMAX_ANGLE)
                latest_angle_deg = round(steer_angle, 1)

                # 8f. Update front state
                robot_status = 'OUT_OF_LANE' if len(lanes_xy_front) == 0 else 'NORMAL'
                if abs(steer_helper.heading_angle) >= math.radians(70):
                    robot_status = 'LARGE_ANGLE'
                    front_stop_conditions.append('Corner Angle too extreme')

                with g.state_lock:
                    g.latest_state['front'] = {
                        'robot_status': robot_status,
                        'FRONT_ANGLE': f'FRONT_ANGLE={latest_angle_deg}',
                        'object_detection': front_detection_output,
                        'stop_conditions': front_stop_conditions,
                    }
                if (vis_front.shape[1] != front.shape[1]) or (vis_front.shape[0] != front.shape[0]):
                    vis_front = cv2.resize(vis_front, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_LINEAR)
                front_display = vis_front

            # --- 9. Rear Camera Processing ---
            if rear is not None:
                vis_rear, t_rear = preprocess_frame(rear, cfg, device)

                # 9a. Lane detection
                with torch.inference_mode():
                    out_rear = lane_model({'img': t_rear})
                    lanes_rear = lane_model.heads.get_lanes(out_rear)[0]

                # 9b. Lane fixing & drawing
                lanes_xy_rear = extract_lane_xy(lanes_rear, cfg, vis_rear.shape)
                lanes_xy_rear = rear_fixer.fix(lanes_xy_rear, frame_width=vis_rear.shape[1])
                draw_lanes(vis_rear, lanes_xy_rear)

                # 9c. Ankle / pose detection
                ankles = get_ankles(vis_rear.copy(), rear_pose)
                ankle_focus.update_frame_size(vis_rear.shape[1], vis_rear.shape[0])
                left_ankle, right_ankle = ankle_focus.focus(ankles)
                draw_ankle_point(vis_rear, left_ankle, (0, 255, 255), 'L ankle')
                draw_ankle_point(vis_rear, right_ankle, (255, 0, 255), 'R ankle')

                # 9d. Rear angle calculation
                midpoint = calculate_midpoint(left_ankle, right_ankle)
                draw_ankle_point(vis_rear, normalize_point(midpoint), (255, 255, 0), 'M')
                angle_rear = calculate_angle_to_center(midpoint, vis_rear)
                visualize_rear(vis_rear, angle_rear)

                feet_detected = midpoint is not None
                rear_servo_val = get_rear_servo(angle_rear, feet_detected)

                # 9e. Build detection output & append stop conditions
                rear_status, _, _ = feet_status(left_ankle, right_ankle, lanes_xy_rear)
                rear_detection_output = build_rear_detection(rear_status)
                if rear_status == 'Left out':
                    rear_stop_conditions.append('Left foot out')
                elif rear_status == 'Right out':
                    rear_stop_conditions.append('Right foot out')
                elif rear_status == 'Both out':
                    rear_stop_conditions.append('Both feet out')
                elif rear_status == 'No feet detected':
                    rear_stop_conditions.append('No feet detected')
                if len(lanes_xy_rear) == 0:
                    rear_stop_conditions.append('No lane Detected')
                    rear_stop_conditions.append('Robot out of lane')

                # 9f. Update rear state
                with g.state_lock:
                    g.latest_state['rear'] = {
                        'status': rear_status,
                        'REAR_ANGLE': f'REAR_ANGLE={rear_servo_val}',
                        'object_detection': rear_detection_output,
                        'stop_conditions': rear_stop_conditions,
                    }
                if (vis_rear.shape[1] != rear.shape[1]) or (vis_rear.shape[0] != rear.shape[0]):
                    vis_rear = cv2.resize(vis_rear, (rear.shape[1], rear.shape[0]), interpolation=cv2.INTER_LINEAR)
                rear_display = vis_rear

            combined_stop_conditions = front_stop_conditions + rear_stop_conditions

            # --- 10. Hold-Time Stop Condition Evaluation ---
            # A condition must be continuously active for its hold duration before STOP fires.
            FRONT_HOLD = {
                'If object is in lane':     HOLD_OBJECT_IN_LANE_SEC,
                'Corner Angle too extreme': HOLD_CORNER_ANGLE_SEC,
                'No lane Detected':         HOLD_FRONT_NO_LANE_SEC,
                'Robot out of lane':        HOLD_FRONT_OUT_LANE_SEC,
            }
            REAR_HOLD = {
                'Left foot out':    HOLD_LEFT_FOOT_SEC,
                'Right foot out':   HOLD_RIGHT_FOOT_SEC,
                'Both feet out':    HOLD_BOTH_FEET_SEC,
                'No feet detected': HOLD_NO_FEET_SEC,
                'No lane Detected': HOLD_REAR_NO_LANE_SEC,
                'Robot out of lane':HOLD_REAR_OUT_LANE_SEC,
            }

            now = time.time()
            actuator_stop_conditions = []
            active_keys = set()

            for c, hold in FRONT_HOLD.items():
                if c in front_stop_conditions:
                    key = f'front:{c}'
                    active_keys.add(key)
                    if key not in g.condition_since:
                        g.condition_since[key] = now
                    elif (now - g.condition_since[key]) >= hold:
                        actuator_stop_conditions.append(c)

            for c, hold in REAR_HOLD.items():
                if c in rear_stop_conditions:
                    key = f'rear:{c}'
                    active_keys.add(key)
                    if key not in g.condition_since:
                        g.condition_since[key] = now
                    elif (now - g.condition_since[key]) >= hold:
                        actuator_stop_conditions.append(f'Rear: {c}')

            # Clear timers for conditions no longer active this frame
            for key in list(g.condition_since.keys()):
                if key not in active_keys:
                    del g.condition_since[key]

            # --- 11. Report Warning Conditions ---
            # Inform the app about conditions that have exceeded hold time,
            # but do NOT stop the robot — only End/hold from the app stops it.
            if len(actuator_stop_conditions) > 0:
                print(f"[WARNING] Conditions active: {actuator_stop_conditions}")
                with g.state_lock:
                    g.latest_state['auto_stop_reason'] = actuator_stop_conditions
            else:
                with g.state_lock:
                    g.latest_state['auto_stop_reason'] = []

            # --- 12. Send Steering Commands ---
            # Forward angle commands to Pi only if still running after condition check.
            if g.is_running:
                send_angles_to_pi(latest_angle_deg, rear_servo_val)

            # --- 13. Broadcast Status & Display ---
            # Push state update at steady cadence and render CV windows.
            with g.state_lock:
                g.latest_state['running'] = g.is_running
            broadcast_status(force=False)

            cv2.putText(front_display, 'MODE: RUNNING CV', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(rear_display,  'MODE: RUNNING CV', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Front AI Camera", front_display)
            cv2.imshow("Rear Backup Camera", rear_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            # --- 14. Idle Preview Mode ---
            # Show raw feeds on laptop before START command is received.
            cv2.putText(front_display, 'MODE: IDLE (RAW)', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(rear_display,  'MODE: IDLE (RAW)', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Front AI Camera", front_display)
            cv2.imshow("Rear Backup Camera", rear_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()