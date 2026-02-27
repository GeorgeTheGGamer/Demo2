"""Provides functions to fit lane lines and calculate the centerline
 for steering angle estimation."""
import math
import numpy as np
import cv2

MIN_POINTS = 3

class SteeringHelper:

    def __init__(self, lanes_xy, frame_width, y_min=100, y_max=300, n_samples=20, idx1=3, idx2=10, threshold=0.1):
        """
        lanes_xy   : list of N lanes, each lane is a list of (x, y) tuples, sorted left→right by x.
                     Can also be a 2-tuple (left_pts, right_pts) for backward compatibility.
        frame_width: width of the frame in pixels, used to find the lane pair closest to center.
        The two lanes whose average-x straddles the frame centre are selected as left/right boundary.
        self.worked is False when fewer than 2 valid lanes are found.
        """
        self.worked = False
        self.heading_angle = 0.0
        self.center_points = []
        self.left_points = []
        self.right_points = []

        if not lanes_xy or len(lanes_xy) < 2:
            return

        # Normalise input
        # Accept both a plain 2-tuple and a list of N lanes
        if len(lanes_xy) == 2 and not isinstance(lanes_xy[0][0], (list, tuple)):
            # Old API: points_xy = (left_pts, right_pts)
            candidates = list(lanes_xy)
        else:
            candidates = list(lanes_xy)

        # Filter out lanes with too few points
        candidates = [lane for lane in candidates if len(lane) >= MIN_POINTS]
        if len(candidates) < 2:
            return

        # Pick the lane pair that straddles the frame centre
        cx = frame_width / 2.0

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
        self.y_min      = y_min
        self.y_max      = y_max
        self.n_samples  = n_samples
        self.idx1       = idx1
        self.idx2       = idx2
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
        ys = np.linspace(self.y_min, self.y_max, self.n_samples)
        x_left  = np.polyval(self.left_coef, ys)
        x_right = np.polyval(self.right_coef, ys)
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
        # Guard against out-of-range indices
        i1 = min(self.idx1, n - 2)
        i2 = min(self.idx2, n - 1)
        x1, y1 = self.center_points[i1]
        x2, y2 = self.center_points[i2]
        dx = x2 - x1
        dy = y2 - y1
        # Negate both components so the vector points toward the horizon (forward).
        # In image coords: right turn → dx > 0 (x drifts right going away from car)
        # → -dx < 0 → theta < 0 (right turn negative, left turn positive).
        theta = math.atan2(-dx, -dy)
        return theta if abs(theta) > self.threshold else 0.0

    def visualization(self, frame):
        """
        Visualize centerline and heading angle on the frame.
        frame: input image to plot on
        """
        img = frame.copy()

        if len(self.center_points) > 1:
            # Draw centerline connect
            pts = np.array(self.center_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)

        # Draw heading angle
        h, w = img.shape[:2]
        cx, cy = int(w / 2), h  # Start from bottom center
        length = 100

        # Calc end points of the heading line based on the angle
        # positive theta = left turn → end_x moves left (cx decreases)
        end_x = int(cx + length * math.sin(self.heading_angle))
        end_y = int(cy - length * math.cos(self.heading_angle))

        cv2.line(img, (cx, cy), (end_x, end_y), (0, 0, 255), 4)

        # Display Angle Text
        angle_deg = math.degrees(self.heading_angle)
        cv2.putText(img, f"Steering: {angle_deg:.2f} deg", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img
