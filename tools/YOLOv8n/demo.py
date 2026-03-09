import cv2
from ultralytics import YOLO
import objectDetector as oD
import poseDetector as pD


def draw_keypoint(img, point, label, color):
    x, y = map(int, point)

    # Outer white circle (contrast)
    cv2.circle(img, (x, y), 8, (255, 255, 255), -1)

    # Inner colored circle
    cv2.circle(img, (x, y), 4, color, -1)

    # Label background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        img,
        (x + 6, y - h - 8),
        (x + 6 + w + 4, y - 2),
        (0, 0, 0),
        -1
    )

    # Label text
    cv2.putText(
        img,
        label,
        (x + 8, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


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


def calculate_angle_to_center(midpoint, frame_width, frame_height):
    """
    Calculate the angle of the midpoint relative to the screen center.
    :param midpoint: midpoint coordinate [x, y]
    :param frame_width: width of the frame
    :param frame_height: height of the frame
    :return: dictionary with angle_x (horizontal), angle_y (vertical), and distance info
    """
    if midpoint is None:
        return None

    # Calculate screen center
    center_x = frame_width / 2
    center_y = frame_height / 2

    # Calculate offset from center
    offset_x = midpoint[0] - center_x
    offset_y = midpoint[1] - center_y

    # Calculate angles in degrees
    # Horizontal angle (pan): positive = right, negative = left
    angle_x = (offset_x / center_x) * 90  # Maps to ±90 degrees max

    # Vertical angle (tilt): positive = down, negative = up
    angle_y = (offset_y / center_y) * 90  # Maps to ±90 degrees max

    # Distance from center (in pixels)
    distance = (offset_x**2 + offset_y**2) ** 0.5

    return {
        'angle_x': angle_x,      # Horizontal angle (pan)
        'angle_y': angle_y,      # Vertical angle (tilt)
        'offset_x': offset_x,    # Horizontal distance from center
        'offset_y': offset_y,    # Vertical distance from center
        'distance': distance,    # Total distance from center
        'midpoint': midpoint
    }


def main():

    # Choose model
    model_o = YOLO("../../checkpoints/yolov8n_int8.tflite")
    model_p = YOLO("../../checkpoints/yolov8n-pose_int8.tflite")

    # Start capturing
    cap = cv2.VideoCapture(0)

    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        obs = frame
        if frame_count % 1 == 0:
            objects = model_o(frame, verbose = False)[0]
            obs = objects.plot()
            l_ankle, r_ankle = pD.get_ankle(frame, model_p)

            # draw ankles (only if found)
            if l_ankle is not None:
                print(l_ankle)
                draw_keypoint(obs, l_ankle, "left" ,(255, 0, 0))

            if r_ankle is not None:
                draw_keypoint(obs, r_ankle, "right", (0, 0, 255))

            # Calculate and draw midpoint
            midpoint = calculate_midpoint(l_ankle, r_ankle)
            if midpoint is not None:
                print(f"Midpoint: {midpoint}")
                draw_keypoint(obs, midpoint, "midpoint", (0, 255, 0))

                # Calculate angle to screen center
                angle_info = calculate_angle_to_center(midpoint, frame_width, frame_height)
                if angle_info is not None:
                    angle_x = angle_info['angle_x']
                    angle_y = angle_info['angle_y']
                    distance = angle_info['distance']

                    print(f"Angle X (Pan): {angle_x:.2f}° | Angle Y (Tilt): {angle_y:.2f}° | Distance: {distance:.2f}px")

                    # Draw angle information on frame
                    info_text = f"Pan: {angle_x:.1f}° | Tilt: {angle_y:.1f}°"
                    cv2.putText(obs, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw center crosshair
                    center_x = int(frame_width / 2)
                    center_y = int(frame_height / 2)
                    cv2.circle(obs, (center_x, center_y), 10, (255, 255, 0), 2)
                    cv2.line(obs, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 0), 2)
                    cv2.line(obs, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        cv2.imshow("captured", obs)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
