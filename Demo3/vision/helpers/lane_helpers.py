"""Helper functions handling lane data"""



def extract_lane_xy(lanes, cfg, frame_shape):
    """Convert lane polylines to lists of (x, y) coordinates"""
    lanes_xy = []
    h, w = frame_shape[:2]
    for lane in lanes:
        pts = lane.to_array(cfg)
        xy = []
        for p in pts:
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h:
                xy.append((x, y))
        if len(xy) >= 2:
            lanes_xy.append(xy)
    lanes_xy.sort(key=lambda xys: xys[0][0])
    return lanes_xy

def interpolate_x_at_y(polyline, y):
    """Given a polyline (list of (x, y) points) and a y-coordinate, return the interpolated x-coordinate at that y."""
    for i in range(1, len(polyline)):
        x1, y1 = polyline[i - 1]
        x2, y2 = polyline[i]
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        if y_min <= y <= y_max and y1 != y2:
            t = (y - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)
    return None

def is_point_in_lane(pt, lanes_xy):
    """Check if a point (x, y) is within the lane boundaries defined by lanes_xy."""
    if pt is None or len(lanes_xy) < 2:
        return False
    cx, cy = pt
    xs_at_y = []
    for lane_xy in lanes_xy:
        x = interpolate_x_at_y(lane_xy, cy)
        if x is not None:
            xs_at_y.append(x)
    if len(xs_at_y) < 2:
        return False
    xs_at_y.sort()
    for i in range(0, len(xs_at_y) - 1):
        if xs_at_y[i] <= cx <= xs_at_y[i + 1]:
            return True
    return False

def normalize_point(pt):
    """Convert a point from tensor to (x, y) tuple of floats, or return None if input is None or undetected.
    YOLO returns [0.0, 0.0] for keypoints it cannot localise — treat these as missing.
    """
    if pt is None:
        return None
    x, y = float(pt[0]), float(pt[1])
    if x == 0.0 and y == 0.0:
        return None  # Undetected YOLO keypoint — not a real position
    return x, y


