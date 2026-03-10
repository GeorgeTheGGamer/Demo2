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

