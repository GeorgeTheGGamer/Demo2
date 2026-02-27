"""
lane_detector.py - Real-Time Lane Detection using OpenCV
Based on: https://medium.com/@nirmalchathura/real-time-lane-detection-with-opencv-and-python
Uses: Gaussian blur, HSV color filtering, Canny edge detection, ROI masking, Hough Transform
"""

import cv2
import numpy as np
from collections import deque


class LaneDetector:
    """Real-time lane detection using classical computer vision techniques."""

    def __init__(self, history_size=5):
        """Initialize the lane detector.

        Args:
            history_size (int): Number of frames to average for smoothing.
        """
        self.left_history = deque(maxlen=history_size)
        self.right_history = deque(maxlen=history_size)

        # HSV thresholds for white lane markings
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 60, 255])

        # HSV thresholds for yellow lane markings (optional)
        self.lower_yellow = np.array([15, 80, 100])
        self.upper_yellow = np.array([35, 255, 255])

        # Canny edge detection thresholds
        self.canny_low = 75
        self.canny_high = 150

        # Hough transform parameters
        self.hough_threshold = 50
        self.hough_min_line_length = 30
        self.hough_max_line_gap = 30

        # Angle filtering (in degrees) - widen range to capture curved lane segments
        self.min_angle = 15
        self.max_angle = 80

    def preprocess(self, frame):
        """Apply Gaussian blur to reduce noise."""
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def color_filter(self, frame):
        """Filter for white and yellow lane markings using HSV color space. """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Detect white lanes
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        # Detect yellow lanes
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        return combined_mask

    def detect_edges(self, mask):
        """Apply Canny edge detection."""
        return cv2.Canny(mask, self.canny_low, self.canny_high)

    def apply_roi(self, edges, frame_shape):
        """Apply Region of Interest mask to focus on lane area.
        Uses a trapezoid: wide at the bottom, narrow at the top."""
        height, width = frame_shape[:2]

        # Trapezoid: bottom spans 10%~90% of width, top spans 40%~60% at 55% height
        roi = np.array([
            [
                (int(width * 0.1), height),           # Bottom-left
                (int(width * 0.4), int(height * 0.55)),  # Top-left
                (int(width * 0.6), int(height * 0.55)),  # Top-right
                (int(width * 0.9), height)            # Bottom-right
            ]
        ], dtype=np.int32)

        # Create mask
        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, roi, 255)

        # Apply mask to edges
        return cv2.bitwise_and(edges, roi_mask)

    def detect_lines(self, roi_edges):
        """Detect lines using Hough Line Transform."""
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        return lines

    def filter_lines_by_angle(self, lines, frame_width):
        """Filter and classify lines as left or right lane based on angle.

        Args:
            lines (np.ndarray): Detected lines from Hough transform.
            frame_width (int): Width of the frame.

        Returns:
            tuple: (left_lines, right_lines) lists.
        """
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        mid_x = frame_width // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle from horizontal
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Filter by angle (ignore near-horizontal lines)
            if angle < self.min_angle or angle > self.max_angle:
                continue

            # Calculate slope
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

            # Classify as left or right based on position and slope
            # Left lanes: negative slope (in image coords), on left side
            # Right lanes: positive slope, on right side
            center_x = (x1 + x2) / 2

            if slope < 0 and center_x < mid_x:
                left_lines.append(line[0])
            elif slope > 0 and center_x > mid_x:
                right_lines.append(line[0])

        return left_lines, right_lines

    def average_lines(self, lines, frame_height):
        """Average multiple line segments into a single lane line.

        Args:
            lines (list): List of line segments.
            frame_height (int): Height of the frame.

        Returns:
            list: List of (x, y) points representing the lane.
        """
        if not lines:
            return []

        # Fit a polynomial to all line points
        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) < 2:
            return []

        try:
            # Fit a 2nd degree polynomial (parabola) to capture curved lanes
            deg = 2 if len(x_coords) >= 3 else 1
            poly = np.polyfit(y_coords, x_coords, deg)
            poly_fn = np.poly1d(poly)

            # Generate points along the lane
            y_start = frame_height
            y_end = int(frame_height * 0.5)

            # Sample 20 points for a smoother curve
            y_values = np.linspace(y_start, y_end, 20)
            points = [(int(poly_fn(y)), int(y)) for y in y_values]

            return points

        except (np.RankWarning, np.linalg.LinAlgError):
            return []

    def smooth_lane(self, current_points, history):
        """Smooth lane detection using temporal averaging.

        Args:
            current_points (list): Current frame's lane points.
            history (deque): Historical lane points.

        Returns:
            list: Smoothed lane points.
        """
        if not current_points:
            # Use last known lane if available
            if history:
                return list(history[-1])
            return []

        history.append(current_points)

        if len(history) < 2:
            return current_points

        # Average across history
        num_points = len(current_points)
        averaged_points = []

        for i in range(num_points):
            x_sum, y_sum = 0, 0
            count = 0

            for hist_pts in history:
                if i < len(hist_pts):
                    x_sum += hist_pts[i][0]
                    y_sum += hist_pts[i][1]
                    count += 1

            if count > 0:
                averaged_points.append((int(x_sum / count), int(y_sum / count)))

        return averaged_points

    def detect(self, frame):
        """Main lane detection pipeline.

        Args:
            frame (np.ndarray): Input BGR frame.

        Returns:
            tuple: (left_lane_points, right_lane_points) lists of (x, y) tuples.
        """
        height, width = frame.shape[:2]

        # 1. Preprocess with Gaussian blur
        blurred = self.preprocess(frame)

        # 2. Color filtering for lane markings
        color_mask = self.color_filter(blurred)

        # 3. Edge detection
        edges = self.detect_edges(color_mask)

        # 4. Apply ROI mask
        roi_edges = self.apply_roi(edges, frame.shape)

        # 5. Detect lines with Hough Transform
        lines = self.detect_lines(roi_edges)

        # 6. Filter and classify lines
        left_lines, right_lines = self.filter_lines_by_angle(lines, width)

        # 7. Average lines for each lane
        left_points = self.average_lines(left_lines, height)
        right_points = self.average_lines(right_lines, height)

        # 8. Temporal smoothing
        left_points = self.smooth_lane(left_points, self.left_history)
        right_points = self.smooth_lane(right_points, self.right_history)

        return left_points, right_points

    def draw_lanes(self, frame, left_points, right_points, fill_lane=True):
        """Draw detected lanes on frame.

        Args:
            frame (np.ndarray): Input BGR frame.
            left_points (list): Left lane points.
            right_points (list): Right lane points.
            fill_lane (bool): Whether to fill the lane area.

        Returns:
            np.ndarray: Frame with lanes drawn.
        """
        overlay = frame.copy()

        # Fill lane area if both lanes detected
        if fill_lane and left_points and right_points:
            # Create polygon from both lane boundaries
            lane_polygon = np.array(left_points + right_points[::-1], dtype=np.int32)
            cv2.fillPoly(overlay, [lane_polygon], (0, 200, 0))  # Green fill
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw left lane (blue)
        if left_points and len(left_points) >= 2:
            pts = np.array(left_points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (255, 0, 0), 3)

        # Draw right lane (red)
        if right_points and len(right_points) >= 2:
            pts = np.array(right_points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 0, 255), 3)

        return frame


# Global instance for easy import
_detector = None


def get_detector():
    """Get or create the global lane detector instance."""
    global _detector
    if _detector is None:
        _detector = LaneDetector()
    return _detector


def detect_lanes(frame):
    """Convenience function for lane detection."""
    detector = get_detector()
    return detector.detect(frame)


def draw_lane_overlay(frame, left_pts, right_pts, fill=True):
    """Draw lane overlay on frame.

    Args:
        frame (np.ndarray): Input BGR frame.
        left_pts (list): Left lane points.
        right_pts (list): Right lane points.
        fill (bool): Whether to fill lane area.

    Returns:
        np.ndarray: Frame with lane overlay.
    """
    detector = get_detector()
    return detector.draw_lanes(frame, left_pts, right_pts, fill)

if __name__ == '__main__':
    # Test the lane detector with webcam
    print("[TEST] Starting lane detection test...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = LaneDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left_pts, right_pts = detector.detect(frame)

        # Draw lanes
        frame = detector.draw_lanes(frame, left_pts, right_pts)

        # Show status
        status = f"Left: {len(left_pts)} pts | Right: {len(right_pts)} pts"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Lane Detection Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
