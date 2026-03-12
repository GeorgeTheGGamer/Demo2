"""YOLO related helper functions for detecting ankles and other objects in frames, and determining feet status."""
from lane_helpers import *

def get_ankle(frame, model):
    """
    Return the coordinates of the ankles detected by the YOLO model.
    :param model: used YOLO model
    :param frame: input frame
    :return: left_ankle coordinate, right_ankle coordinate in [x,y] format, or None if no ankle found
    """
    left_ankle = None
    right_ankle = None

    # Run model to get boxes and keypoints
    results = model(frame, verbose=False)[0]
    boxes = results.boxes
    keypoints = results.keypoints

    # Filter person by box size, bigger box size, higher probability
    max_size = 0
    for i, box in enumerate(boxes):
        # Get box details
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)
        size = x2 - x1
        if conf > 0.5 and size > max_size:
            max_size = size
            # 15 for left ankle, 16 for right ankle
            left_ankle = keypoints.xy[i][15]
            right_ankle = keypoints.xy[i][16]

    return left_ankle, right_ankle

def get_ankles(frame, model):
    """
    Return the coordinates of the ankles detected by the YOLO model.
    :param model: used YOLO model
    :param frame: input frame
    :return: list of (left_ankle, right_ankle) coordinates in [x,y] format, or empty list if no ankle found
    """
    ankles = []

    # Run model to get boxes and keypoints
    results = model(frame, verbose=False)[0]
    boxes = results.boxes
    keypoints = results.keypoints

    for i, box in enumerate(boxes):
        # Get box details
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)
        if conf > 0.5:
            left_ankle = keypoints.xy[i][15]
            right_ankle = keypoints.xy[i][16]
            ankles.append((left_ankle, right_ankle))

    return ankles

def feet_status(left_ankle, right_ankle, lanes_xy):
    left = normalize_point(left_ankle)
    right = normalize_point(right_ankle)
    left_in = False if left is None else is_point_in_lane(left, lanes_xy)
    right_in = False if right is None else is_point_in_lane(right, lanes_xy)
    if left is None and right is None:
        return 'No feet detected', left_in, right_in
    if left_in and right_in:
        return 'Safe', left_in, right_in
    if (not left_in) and right_in:
        return 'Left out', left_in, right_in
    if left_in and (not right_in):
        return 'Right out', left_in, right_in
    return 'Both out', left_in, right_in

def get_objects(frame, model, conf_thres=0.3):
    """
    Read objects from a frame, return boxes details
    :param model: used YOLO model
    :param frame: input frame
    :param conf_thres: threshold of confidence rate for detecting objects
    :return: all details in a dictionary {cls,conf,bbox}
    """
    results = model(frame, verbose=False)[0]
    res_list = []
    for box in results.boxes:
        # get box details
        cls = int(box.cls[0].item())
        conf = float(box.conf)
        if conf < conf_thres:
            continue

        x1,y1,x2,y2 = box.xyxy[0].tolist()
        res_list.append({
            "cls": cls,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
        })
    return res_list

def is_object_close_to_lane(obj, lanes_xy, distance_px=80):
    if len(lanes_xy) == 0:
        return False
    x1, y1, x2, y2 = obj['bbox']
    sample_points = [
        ((x1 + x2) / 2.0, y2),
        (x1, y2),
        (x2, y2),
    ]
    for cx, cy in sample_points:
        lane_xs = []
        for lane_xy in lanes_xy:
            lx = interpolate_x_at_y(lane_xy, cy)
            if lx is not None:
                lane_xs.append(lx)
        if len(lane_xs) == 0:
            continue
        min_dist = min(abs(cx - lx) for lx in lane_xs)
        if min_dist <= distance_px:
            return True
    return False

def build_front_detection(objects, lanes_xy, frame_shape, names=None, close_ratio=0.7):
    warning = []
    danger = []
    h, w = frame_shape[:2]
    frame_area = max(1.0, float(h * w))

    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        in_lane = (
            is_point_in_lane((x1, y2), lanes_xy)
            or is_point_in_lane(((x1 + x2) / 2.0, y2), lanes_xy)
            or is_point_in_lane((x2, y2), lanes_xy)
        )
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        is_close = ((box_w * box_h) / frame_area) >= close_ratio

        cls_id = obj['cls']
        raw_name = names[cls_id] if (names is not None and cls_id in names) else str(cls_id)
        name = 'person' if raw_name == 'person' else 'object'
        side = 'left' if ((x1 + x2) / 2.0) < (w / 2.0) else 'right'
        label = f'{name}({side})'

        if in_lane:
            danger.append(label)
        elif is_close:
            warning.append(label)

    return {'warning': warning, 'danger': danger}


def build_rear_detection(status):
    warning = []
    danger = []
    if status == 'No feet detected':
        warning.append('person(unknown_ankles)')
    elif status == 'Left out':
        danger.append('left_foot(out_of_lane)')
    elif status == 'Right out':
        danger.append('right_foot(out_of_lane)')
    elif status == 'Both out':
        danger.append('left_foot(out_of_lane)')
        danger.append('right_foot(out_of_lane)')
    return {'warning': warning, 'danger': danger}

def calculate_midpoint(left_ankle, right_ankle):
    """
    Calculate the midpoint between left and right ankles.
    :return: midpoint coordinate [x, y] or None if either ankle is None
    """
    if left_ankle is None or right_ankle is None:
        return None

    x_mid = (left_ankle[0] + right_ankle[0]) / 2
    y_mid = (left_ankle[1] + right_ankle[1]) / 2

    return [x_mid, y_mid]