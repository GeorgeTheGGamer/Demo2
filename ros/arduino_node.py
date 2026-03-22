import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

class ArduinoNode(Node):
    def __init__(self):
        super().__init__('arduino_node')

        self.subscription = self.create_subscription(String, "robot/motor_commands", self.command_callback, 10)

        self.serial_port = SERIAL_PORT
        self.baud_rate = BAUD_RATE

        self.robot_running = False

        self.stop_requested = False

        self.read_timer = self.create_timer(0.1, self.read_from_arduino)

        self.enforce_stop_timer = self.create_timer(0.5, self.enforce_stop)

        try:
            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)
            self.get_logger().info(f"Connected to arduino on port {self.serial_port}")
        except Exception as e:
            self.get_logger().error("Failed to connect to arduino")
            self.arduino = None

    def command_callback(self, msg):
        command_str = msg.data
        if command_str == "START":
            self.robot_running = False  # reset until we get confirmation
            self.stop_requested = False
        if command_str == "STOP":
            self.stop_requested = True

        # Always send START and STOP
        # Only send angles if robot confirmed running
        if command_str.startswith("FRONT_ANGLE") and not self.robot_running:
            self.get_logger().warn("Skipping angle command, robot not confirmed running")
            return

        if self.arduino:
            try:
                self.arduino.write(f"{command_str}\n".encode('utf-8'))
                self.arduino.flush()
            except Exception as e:
                self.get_logger().error(f"Failed to write to serial: {e}")
        else:
            self.get_logger().error("Arduino not set up")

    def enforce_stop(self):

        if self.stop_requested and self.robot_running:
            self.get_logger().warn("Robot not confirmed stop yet. Resending STOP command")
            if self.arduino:
                try:
                    self.arduino.write("STOP\n".encode('utf-8'))
                    self.arduino.flush()
                except Exception as e:
                    self.get_logger().error(f"Failed to write STOP to serial: {e}")
        
        elif not self.robot_running and self.stop_requested:
            self.stop_requested = False
            self.get_logger().info("Stop confirmed, ending STOP enforcements.")

    def read_from_arduino(self):

        if self.arduino and self.arduino.in_waiting > 0:
            try:
                arduino_reply = self.arduino.readline().decode('utf-8').strip()
                if arduino_reply:
                    self.get_logger().info(f"ARDUINO SAYS: {arduino_reply}")
                    if arduino_reply == "1":
                        self.robot_running = True
                    if arduino_reply == "0":
                        self.robot_running = False
                        self.stop_requested = False

            except Exception as e:
                self.get_logger().error(f"Failed to read from Arduino: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArduinoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.arduino:
            node.arduino.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()