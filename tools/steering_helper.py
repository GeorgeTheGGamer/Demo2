"""Provides functions to fit lane lines and calculate the centerline
 for steering angle estimation."""
import math
import numpy as np
import cv2

MIN_POINTS = 3

class SteeringHelper:

    def __init__(self, points_xy, y_min=100, y_max=300, n_samples=20, idx1=3, idx2=10, threshold=0.1):
        """ Validation: points_xy should be a tuple of (left_points, right_points), where each is a list of (x, y).
        number of points in each lane should be >= MIN_POINTS to fit a curve. Otherwise, self.worked will be False."""

        self.worked = False
        self.heading_angle = 0.0
        self.center_points = []
        if not points_xy or len(points_xy) < 2:
            return

        self.left_points, self.right_points = points_xy[0], points_xy[1]
        if (len(self.left_points) >= MIN_POINTS and
                len(self.right_points) >= MIN_POINTS):
            self.y_min = y_min # sampling range in y direction, can be adjusted based on the camera view
            self.y_max = y_max
            self.n_samples = n_samples # number of points to sample for centerline, can be adjusted for smoother or more detailed centerline
            self.idx1 = idx1 # indices of points to calculate heading, can be adjusted based on the expected curvature of the lane
            self.idx2 = idx2
            self.threshold = threshold # threshold for heading angle, can be adjusted to filter out small angles that may be noise
            self.left_coef = self.poly_fit(self.left_points)
            self.right_coef = self.poly_fit(self.right_points)
            self.center_points = self.sample_centerline()
            self.heading_angle = self.heading_from_centerline()
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
        idx1, idx2: indices of points to calculate heading
        return: heading angle in radians, where 0 means straight ahead,
        positive is left turn, negative is right turn
        if angle is smaller than threshold, return 0 to avoid noise
        """
        x1, y1 = self.center_points[self.idx1]
        x2, y2 = self.center_points[self.idx2]
        dx, dy = (x2 - x1), (y2 - y1)
        theta = math.atan2(dx, dy)
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
        end_x = int(cx - length * math.sin(self.heading_angle))
        end_y = int(cy - length * math.cos(self.heading_angle))

        cv2.line(img, (cx, cy), (end_x, end_y), (0, 0, 255), 4)

        # Display Angle Text
        angle_deg = math.degrees(self.heading_angle)
        cv2.putText(img, f"Steering: {angle_deg:.2f} deg", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img
