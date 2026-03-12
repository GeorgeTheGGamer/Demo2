"""opencv and torch based frame processing for lane detection and visualization"""
import math

import cv2
import numpy as np
import torch

from Demo3.vision.helpers.lane_helpers import is_point_in_lane


def preprocess_frame(frame, cfg, device):
    frame = cv2.resize(frame, (cfg.ori_img_w, cfg.ori_img_h), interpolation=cv2.INTER_LINEAR)
    cropped = frame[cfg.cut_height:, :, :]
    resized = cv2.resize(cropped, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return frame, tensor

def draw_lanes(frame, lanes_xy, line_width=4):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    for i, xy in enumerate(lanes_xy):
        if len(xy) < 2:
            continue
        color = colors[i % len(colors)]
        for j in range(1, len(xy)):
            cv2.line(frame, xy[j - 1], xy[j], color, thickness=line_width)

def make_placeholder_frame(title, width=960, height=540):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, title, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(frame, 'Waiting for stream...', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    return frame

def draw_front_objects(frame, objects, lanes_xy, names=None):
    for obj in objects:
        if 'bbox' not in obj:
            continue
        x1, y1, x2, y2 = obj['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(obj.get('conf', 0.0))
        in_lane = (
            is_point_in_lane((x1, y2), lanes_xy)
            or is_point_in_lane(((x1 + x2) / 2.0, y2), lanes_xy)
            or is_point_in_lane((x2, y2), lanes_xy)
        )

        # Only display if any of the 3 bottom lane-contact points are in lane.
        if not in_lane:
            continue

        color = (0, 0, 255)
        label = 'in_lane'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_ankle_point(frame, pt, color, label):
    if pt is None:
        return
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(frame, (x, y), 7, color, -1)
    cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def visualize_rear(frame, angle):
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

def visualize_front(frame, center_points, heading_angle):
    """
    Visualize centerline and heading angle on the frame.
    frame: input image to plot on
    """

    if len(center_points) > 2:
        # Draw centerline points only (no connecting line)
        for (x, y) in center_points:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

    # Draw heading angle
    h, w = frame.shape[:2]
    cx, cy = int(w / 2), h  # Start from bottom center
    length = 100

    # Calc end points of the heading line based on the angle
    # positive theta = left turn → end_x moves left (cx decreases)
    end_x = int(cx + length * math.sin(heading_angle))
    end_y = int(cy - length * math.cos(heading_angle))

    cv2.line(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 4)

    # Display Angle Text
    angle_deg = math.degrees(heading_angle)
    cv2.putText(frame, f"Steering: {angle_deg:.2f} deg", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)