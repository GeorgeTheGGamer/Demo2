"""Provides functions to fit lane lines and calculate the centerline
 for steering angle estimation."""
import math
import time
from typing import Optional

import numpy as np
import Demo3.states.globals as g
from Demo3.config.configs import *

MIN_POINTS = 3

class SteeringHelper:

    def __init__(self, lanes_xy, frame_shape , n_samples=20, threshold=30):
        """
        lanes_xy   : list of N lanes, each lane is a list of (x, y) tuples, sorted left→right by x.
                     Can also be a 2-tuple (left_pts, right_pts) for backward compatibility.
        frame_shape: shapes of the frame in pixels, used to find the lane pair closest to center.
        The two lanes whose average-x straddles the frame center are selected as left/right boundary.
        self.worked is False when fewer than 2 valid lanes are found.
        """
        self.worked = False
        self.heading_angle = 0.0
        self.center_points = []
        self.left_points = []
        self.right_points = []
        self.frame_height, self.frame_width = frame_shape[:2]

        if not lanes_xy:
            return

        candidates = list(lanes_xy)

        # Filter out lanes with too few points
        candidates = [lane for lane in candidates if len(lane) >= MIN_POINTS]
        if len(candidates) < 1:
            return

        # If only one lane detected, use it directly as the centerline
        if len(candidates) == 1:
            self.left_points  = candidates[0]
            self.right_points = candidates[0]
            self.n_samples  = n_samples
            self.left_coef  = self.poly_fit(self.left_points)
            self.right_coef = self.right_coef = self.left_coef
            self.center_points = self.sample_centerline()
            self.heading_angle = self.heading_from_centerline()
            self.worked = True
            return

        # Pick the lane pair that straddles the frame center
        cx = self.frame_width / 2.0

        def mean_x(lane):
            return sum(p[0] for p in lane) / len(lane)

        # Sort candidates by mean-x (should already be sorted, but be safe)
        candidates.sort(key=mean_x)
        mean_xs = [mean_x(lane) for lane in candidates]

        # Find the rightmost lane whose mean-x is still <= cx (left boundary)
        # and the leftmost lane whose mean-x is > cx (right boundary)
        left_idx = None
        right_idx = None
        for i, mx in enumerate(mean_xs):
            if mx <= cx:
                left_idx = i          # keep updating → take the rightmost one ≤ cx
            else:
                if right_idx is None:
                    right_idx = i     # take the leftmost one > cx

        # Fallback: if all lanes are on one side, use the two adjacent centre lanes
        if left_idx is None:
            left_idx, right_idx = 0, 1
        elif right_idx is None:
            left_idx, right_idx = len(candidates) - 2, len(candidates) - 1

        self.left_points  = candidates[left_idx]
        self.right_points = candidates[right_idx]

        # Fit & compute
        self.n_samples  = n_samples
        self.left_coef  = self.poly_fit(self.left_points)
        self.right_coef = self.poly_fit(self.right_points)
        self.center_points  = self.sample_centerline()
        self.heading_angle  = self.heading_from_centerline()
        self.worked = True

    @staticmethod
    def poly_fit(points, degree=2):
        """
        points: list of (x, y) representing lanes
        return: poly coefficients for x = f(y)
        """
        pts = np.asarray(points, dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]

        order = np.argsort(y)
        x, y = x[order], y[order]

        # fit x = a*y^2 + b*y + c (degree=2) or linear (degree=1)
        coef = np.polyfit(y, x, deg=degree)   # coef: [a, b, c] or [b, c]
        return coef

    def sample_centerline(self):
        """
        y_min, y_max: sampling range in y direction
        n_samples: number of points to sample
        return: center_points list of (x, y)
        """
        # Use the actual y range of the lane points to avoid extrapolation
        left_ys  = [p[1] for p in self.left_points]
        right_ys = [p[1] for p in self.right_points]

        # Overlap region: the tighter of the two ranges
        y_lo = max(min(left_ys), min(right_ys))
        y_hi = min(max(left_ys), max(right_ys))

        # Fall back to union if there is no overlap
        if y_lo >= y_hi:
            y_lo = min(min(left_ys), min(right_ys))
            y_hi = max(max(left_ys), max(right_ys))

        ys = np.linspace(y_lo, y_hi, self.n_samples)
        x_left   = np.polyval(self.left_coef, ys)
        x_right  = np.polyval(self.right_coef, ys)
        x_center = (x_left + x_right) / 2.0

        center_points = list(zip(x_center.tolist(), ys.tolist()))
        return center_points

    def heading_from_centerline(self):
        """
        idx1, idx2: indices of points to calculate heading.
        center_points are sampled from y_min (near horizon) to y_max (near car),
        so idx2 > idx1 means the vector points from horizon toward the car.
        To express "forward" direction (car → horizon), we negate the vector.
        return: heading angle in radians, where 0 means straight ahead,
        positive is left turn, negative is right turn.
        if angle is smaller than threshold, return 0 to avoid noise.
        """
        n = len(self.center_points)
        if n < 2:
            return 0.0

        # New logic: use the center bottom point and the top point to calculate the heading angle.
        x1, y1 = self.center_points[-5] # top point (horizon)
        x2, y2 = self.frame_width/2, self.frame_height # bottom point (representing car position)

        dx = x1 - x2
        dy = y1 - y2

        theta = math.atan2(-dx, -dy)
        theta_deg = math.degrees(theta)

        # Old angle to keep car go straight in straight lane
        x3, y3 = self.center_points[-1] # bottom point (car position)
        dx2 = x3 - x2
        dy2 = y3 - y2
        theta2 = math.atan2(-dx2, -dy2)
        theta2_deg = math.degrees(theta2)

        # Priority: 1. angle heading to center of lane 2. keep car go in correct direction 3. 0
        if abs(theta_deg) >= g.current_threshold:
            return theta_deg
        elif abs(theta2_deg) >= g.current_threshold:
            return theta2_deg
        else:
            return 0.0

def angle_deg_to_servo_held(angle_deg: float) -> Optional[float]:
    """
    Appends raw angle to window, then checks if the tail shows an overall
    increasing or decreasing trend by comparing abs(first) vs abs(last).
    Allows gaps/noise in between (e.g. [5,2,10,5,15] counts as increasing).
    If trend detected, returns tail[0] and subtracts it from all window values.
    """
    g.angle_window.append(angle_deg)
    if len(g.angle_window) < STEER_VOTE_WINDOW:
        return None # not enough data yet

    tail = list(g.angle_window)[-STEER_VOTE_THRESHOLD:]

    growing   = sum(1 for v in tail if abs(v) > abs(tail[0]))
    shrinking = sum(1 for v in tail if abs(v) < abs(tail[0]))

    if growing >= STEER_VOTE_THRESHOLD or shrinking >= STEER_VOTE_THRESHOLD:
        result = tail[0]
        for i in range(len(g.angle_window)):
            g.angle_window[i] -= result
        return result

    return None

def rear_angle_to_servo(angle_deg: float) -> int:
    """
    Map rear heading angle (-45 to +45 degrees) to a servo value in 5-degree
    increments between 45 and 135.
    Positive angle = person to the right -> higher servo (towards 135)
    Negative angle = person to the left  -> lower servo  (towards 45)
    90 = centred
    """
    clamped = max(min(angle_deg, 45.0), -45.0)
    raw = 90.0 + clamped
    snapped = round(float(raw) / 5.0) * 5
    return max(45, min(135, snapped))

def get_rear_servo(angle_deg, feet_detected: bool) -> int:
    """
    Returns the rear servo value (45-135 in 5° increments).
    - If feet are detected: compute from angle and update last-seen timestamp.
    - If no feet: hold last position. After REAR_NO_FEET_HOLD_SEC seconds,
      reset to 90° (centre).
    """
    now = time.time()

    if feet_detected:
        servo = rear_angle_to_servo(angle_deg if angle_deg is not None else 0.0)
        g.rear_servo_state['servo'] = servo
        g.rear_servo_state['last_seen'] = now
        return servo
    else:
        # No feet — hold last position until timeout, then centre
        if (now - g.rear_servo_state['last_seen']) >= REAR_NO_FEET_HOLD_SEC:
            g.rear_servo_state['servo'] = 90
        return g.rear_servo_state['servo']

def calculate_angle_to_center(midpoint, frame):
    frame_height, frame_width = frame.shape[:2]
    if midpoint is None:
        return None

    # Vector from midpoint to top-center (frame_width/2, 0)
    center_x = frame_width / 2
    dx = midpoint[0] - center_x
    dy = 0 - midpoint[1]
    angle_x = np.degrees(np.arctan2(dx, -dy))

    return angle_x
