


class FocusHelper:

    def __init__(self, frame_width=640, frame_height=480):
        self.last_coord_pair = None
        self.frame_width = frame_width
        self.frame_height = frame_height

    def update_frame_size(self, frame_width, frame_height):
        """Update the frame size, which may change if the video resolution changes."""
        self.frame_width = frame_width
        self.frame_height = frame_height

    @staticmethod
    def dist(a, b):
        """Calculate the squared distance between two points a and b."""
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def get_midpoint(self, ankle_pair):
        """Get midpoint of (left_ankle, right_ankle) pair for distance calculation."""
        left, right = ankle_pair
        if left is None and right is None:
            return None
        if left is None:
            return right[0], right[1]
        if right is None:
            return left[0], left[1]
        return ((left[0] + right[0]) / 2,
                (left[1] + right[1]) / 2)

    def get_center_ankle(self, ankles):
        """
        ankles: list of (left_ankle, right_ankle) pairs
        Returns the pair whose midpoint is closest to frame center.
        """
        if not ankles:
            return None
        cx, cy = self.frame_width / 2, self.frame_height / 2
        return min(ankles, key=lambda pair: self.dist(
            self.get_midpoint(pair) or (cx, cy), (cx, cy)
        ))

    def focus(self, ankles):
        """
        ankles: list of (left_ankle, right_ankle) pairs
        Returns the (left_ankle, right_ankle) pair closest to last known position.
        """
        if not ankles:
            return self.last_coord_pair

        if self.last_coord_pair is None:
            self.last_coord_pair = self.get_center_ankle(ankles)
        else:
            last_mid = self.get_midpoint(self.last_coord_pair)
            cx, cy = self.frame_width / 2, self.frame_height / 2
            ref = last_mid if last_mid is not None else (cx, cy)
            self.last_coord_pair = min(ankles, key=lambda pair: self.dist(
                self.get_midpoint(pair) or ref, ref
            ))

        return self.last_coord_pair
