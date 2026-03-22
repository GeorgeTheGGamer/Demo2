import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import socket
import threading

SERVER_IP = "192.168.118.173" # laptop ip
REAR_VIDEO_PORT = 8000
FRONT_VIDEO_PORT = 8002
COMMAND_PORT = 8001
MAX_DGRAM = 65507

class BridgeNode(Node):
    def __init__(self):
        super().__init__("bridge_node")

        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_socket.bind(("0.0.0.0", COMMAND_PORT))

        self.bridge = CvBridge()

        self.publisher = self.create_publisher(String, "robot/motor_commands", 10)

        self.rear_video_subscription = self.create_subscription(Image, "robot/rear_camera_feed", lambda msg: self.image_callback(msg, REAR_VIDEO_PORT), 10)

        self.front_video_subscription = self.create_subscription(Image, "robot/front_camera_feed", lambda msg: self.image_callback(msg, FRONT_VIDEO_PORT), 10)

        self.listen_thread = threading.Thread(target=self.listen_command, daemon=True)
        self.listen_thread.start()

        self.get_logger().info(f"Bridge active, streaming to {SERVER_IP}, ports: {REAR_VIDEO_PORT}, {FRONT_VIDEO_PORT}")
        self.get_logger().info(f"Listening for UDP commands on {COMMAND_PORT}")

    def image_callback(self, msg, port_num):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 35])

            if len(buffer) < MAX_DGRAM:
                self.video_socket.sendto(buffer.tobytes(), (SERVER_IP, port_num))

        except Exception as e:
            self.get_logger().error(f"Video send failed: {e}")

    def listen_command(self):
        while True:
            try:
                data, addr = self.cmd_socket.recvfrom(1024)
                command_str = data.decode('utf-8').strip()

                self.get_logger().info(f"Caught Wi-Fi command: '{command_str}'")

                ros_msg = String()
                ros_msg.data = command_str

                self.publisher.publish(ros_msg)

            except Exception as e:
                self.get_logger().error(f"UDP Listener Error: {e}")

    def destroy_node(self):
        self.video_socket.close()
        self.cmd_socket.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()