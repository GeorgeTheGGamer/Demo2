"""Provides functions to fit lane lines and calculate the centerline
 for steering angle estimation."""
import math
import numpy as np
import cv2

MIN_POINTS = 3

class SteeringHelper:

    #TODO: 1. remove threshold while in straight lines
    #TODO: 2. add heading angle for steering
    #TODO: 3. add lane angle and facing angle to keep car go straight in straight lane
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

        if not lanes_xy or len(lanes_xy) < 2:
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
            self.threshold  = threshold
            self.left_coef  = self.poly_fit(self.left_points)
            self.right_coef = self.right_coef = self.left_coef
            self.center_points = self.sample_centerline()
            self.heading_angle = self.heading_from_centerline()
            self.worked = True
            return

        # Pick the lane pair that straddles the frame center
        self.frame_width = frame_shape[1]
        self.frame_height = frame_shape[0]
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
        self.threshold  = threshold
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
        x1, y1 = self.center_points[0] # top point (horizon)
        x2, y2 = self.frame_width/2, self.frame_height # bottom point (representing car position)

        dx = x1 - x2
        dy = y1 - y2

        theta = math.atan2(-dx, -dy)
        theta_deg = math.degrees(theta)

        # Old angle to keep car go straight in straight lane
        # TODO: only use it when on straight lane, otherwise it will cause wrong angle when lane is curved
        x3, y3 = self.center_points[-1] # bottom point (car position)
        dx2 = x3 - x2
        dy2 = y3 - y2
        theta2 = math.atan2(-dx2, -dy2)
        theta2_deg = math.degrees(theta2)

        return theta, theta2_deg if abs(theta_deg) > self.threshold else 0.0

    def visualization(self, frame):
        """
        Visualize centerline and heading angle on the frame.
        frame: input image to plot on
        """

        if len(self.center_points) > 2:
            # Draw centerline points only (no connecting line)
            for (x, y) in self.center_points:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # Draw heading angle
        h, w = frame.shape[:2]
        cx, cy = int(w / 2), h  # Start from bottom center
        length = 100

        # Calc end points of the heading line based on the angle
        # positive theta = left turn → end_x moves left (cx decreases)
        end_x = int(cx + length * math.sin(self.heading_angle))
        end_y = int(cy - length * math.cos(self.heading_angle))

        cv2.line(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 4)

        # Display Angle Text
        angle_deg = math.degrees(self.heading_angle)
        cv2.putText(frame, f"Steering: {angle_deg:.2f} deg", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
