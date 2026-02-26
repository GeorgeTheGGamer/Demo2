"""
person_center_tracker.py - Reusable module for tracking person center from pose estimation.

Computes the center point of a person using shoulder and hip keypoints.
Can be used in any project with pose estimation (MoveNet, etc.).
"""

def compute_person_center(pose_model, keypoints, meta=None):
    """
    Compute the center of a person from pose keypoints.
    
    Args:
        pose_model: Pose estimator object with get_keypoint() method
        keypoints: Raw keypoints from pose model
        meta: Optional metadata dict with 'scale' and 'offset_y' from resize operation
              If None, assumes coordinates are already in original image space
    
    Returns:
        tuple: (center_x, center_y) in original image coordinates, or None if insufficient keypoints
    """
    from resize_utils import map_point_to_original
    
    # Extract shoulder and hip keypoints
    l_shoulder = pose_model.get_keypoint(keypoints, 'left_shoulder', 0.2)
    r_shoulder = pose_model.get_keypoint(keypoints, 'right_shoulder', 0.2)
    l_hip = pose_model.get_keypoint(keypoints, 'left_hip', 0.2)
    r_hip = pose_model.get_keypoint(keypoints, 'right_hip', 0.2)
    
    # Collect torso points
    torso_points = []
    for pt in [l_shoulder, r_shoulder, l_hip, r_hip]:
        if pt:
            if meta:
                # Map from resized space (192x192) back to original
                px, py = pt[0] * 192, pt[1] * 192
                real_pt = map_point_to_original(px, py, meta)
                torso_points.append(real_pt)
            else:
                # Already in original space
                torso_points.append((pt[0], pt[1]))
    
    # Compute center if we have at least 2 keypoints
    if len(torso_points) < 2:
        return None
    
    center_x = sum(p[0] for p in torso_points) / len(torso_points)
    center_y = sum(p[1] for p in torso_points) / len(torso_points)
    
    return (center_x, center_y)


def draw_person_center(frame, center, label="Person Center"):
    """
    Draw a circle and label at the person center on the frame.
    
    Args:
        frame: OpenCV frame to draw on
        center: (x, y) tuple of center coordinates
        label: Optional label text
    
    Returns:
        frame: Modified frame with center drawn
    """
    import cv2
    
    if center is None:
        return frame
    
    x, y = int(center[0]), int(center[1])
    
    # Draw blue circle
    cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
    
    # Draw label
    cv2.putText(frame, label, (x - 50, y - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame


def get_person_center_from_detection(detections, confidence_threshold=0.5):
    """
    Get person center from object detections (bounding boxes).
    Selects the LARGEST detection to avoid picking up feet or partial detections.
    Works at any distance - whether full body or just legs are visible.
    
    Args:
        detections: List of detections from object_detector
                   Each detection: (x, y, w, h, confidence, ...)
        confidence_threshold: Minimum confidence to consider
    
    Returns:
        tuple: (center_x, center_y) or None
    """
    if not detections:
        return None
    
    # Filter by confidence
    valid_detections = [d for d in detections if d[4] > confidence_threshold]
    if not valid_detections:
        return None
    
    # Pick the LARGEST detection (most likely the main person, not feet/noise)
    largest = max(valid_detections, key=lambda d: d[2] * d[3])  # d[2]*d[3] is box area (w*h)
    
    x, y, w, h = largest[:4]
    center_x = x + w / 2
    center_y = y + h / 2
    
    return (center_x, center_y)


def get_lidar_angle(person_center, frame_width, cam_fov_degrees=60):
    """
    Convert person center pixel position to lidar rotation angle.
    Useful for steering a lidar to follow a person.
    
    Args:
        person_center: (x, y) tuple of center coordinates
        frame_width: Width of the frame in pixels
        cam_fov_degrees: Camera field of view in degrees (default 60)
    
    Returns:
        float: Angle in degrees (-90 to +90 for center to edges)
               Negative = left, Positive = right, 0 = center
    """
    if person_center is None:
        return 0
    
    center_x = person_center[0]
    image_center = frame_width / 2
    
    # Normalized position (-1 to +1)
    normalized_offset = (center_x - image_center) / image_center
    
    # Convert to angle
    half_fov = cam_fov_degrees / 2
    angle = normalized_offset * half_fov
    
    return angle
