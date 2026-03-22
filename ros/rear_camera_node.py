import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

CAMERA_DEVICE = "/dev/video0"

class RearCameraNode(Node):
    def __init__(self):
        super().__init__('rear_camera_node')

        self.publisher = self.create_publisher(Image, 'robot/rear_camera_feed', 10)

        self.cap = cv2.VideoCapture(CAMERA_DEVICE)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.bridge = CvBridge()

        self.timer = self.create_timer(0.033, self.cap_and_pub)

        self.get_logger().info("Rear camera node active. Publishing frames...")

    def cap_and_pub(self):

        success, frame = self.cap.read()

        if success:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            self.publisher.publish(ros_image)

        else:
            self.get_logger().warning("Dropped frame")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    cam_node = RearCameraNode()

    try:
        rclpy.spin(cam_node)
    except KeyboardInterrupt:
        pass
    finally:
        cam_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()