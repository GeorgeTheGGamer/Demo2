import cv2
import time
import socket
import threading
import importlib

try:
    serial = importlib.import_module('serial')
except Exception:
    serial = None

# -----------------------------
# Network + Camera Config
# -----------------------------
LAPTOP_IP = '192.168.8.173'  # your laptop IP
FRONT_PORT = 8000
REAR_PORT = 8002
CMD_PORT = 8001
MAX_DGRAM = 65507

FRONT_CAMERA_DEVICE = '/dev/video0'
REAR_CAMERA_DEVICE = '/dev/video2'

FRAME_W = 320
FRAME_H = 240
FRAME_FPS = 30
JPEG_QUALITY = 35

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200


class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.stream.set(cv2.CAP_PROP_FPS, FRAME_FPS)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                time.sleep(0.01)
                continue
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.stream.release()
        except Exception:
            pass


class PiBridge:
    def __init__(self):
        self.running = True
        self.streaming_enabled = False
        self.stream_lock = threading.Lock()

        self.tx_front = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tx_rear = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Unified Command and Status Listener
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind(('0.0.0.0', CMD_PORT))

        self.arduino = self._connect_arduino()

        self.front_cam = CameraStream(FRONT_CAMERA_DEVICE).start()
        self.rear_cam = CameraStream(REAR_CAMERA_DEVICE).start()

    def _connect_arduino(self):
        if serial is None:
            print('[PI] pyserial not installed; running without Arduino link.')
            return None
        try:
            arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            print(f'[PI] ✅ Connected to Arduino on {SERIAL_PORT}')
            return arduino
        except Exception as e:
            print(f'[PI] ⚠️ Arduino serial unavailable: {e}')
            return None

    def _write_arduino(self, msg):
        print(f'[PI] -> ARDUINO: {msg}')
        if self.arduino:
            try:
                self.arduino.write(f'{msg}\n'.encode('utf-8'))
            except Exception:
                pass

    def _set_streaming(self, enabled):
        with self.stream_lock:
            self.streaming_enabled = enabled
        print(f"[PI] Streaming {'ENABLED' if enabled else 'DISABLED'}")

    def _can_stream(self):
        with self.stream_lock:
            return self.streaming_enabled

    def _stream_loop(self):
        print(f'[PI] Front stream -> {LAPTOP_IP}:{FRONT_PORT}')
        print(f'[PI] Rear stream  -> {LAPTOP_IP}:{REAR_PORT}')

        while self.running:
            if not self._can_stream():
                time.sleep(0.02)
                continue

            front = self.front_cam.read()
            rear = self.rear_cam.read()

            if front is not None:
                ok_f, buf_f = cv2.imencode('.jpg', front, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok_f and len(buf_f) <= MAX_DGRAM:
                    self.tx_front.sendto(buf_f, (LAPTOP_IP, FRONT_PORT))

            if rear is not None:
                ok_r, buf_r = cv2.imencode('.jpg', rear, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok_r and len(buf_r) <= MAX_DGRAM:
                    self.tx_rear.sendto(buf_r, (LAPTOP_IP, REAR_PORT))

    def _unified_listener(self):
        print(f'[PI] 🎧 Listening for System Commands and CV Control on port {CMD_PORT}')
        while self.running:
            try:
                data, _ = self.cmd_sock.recvfrom(1024)
                msg = data.decode('utf-8', errors='ignore').strip()
                if not msg:
                    continue

                # 1. Steering command from laptop (plain text)
                if msg.startswith("ANGLE="):
                    angle_value = msg.split("=", 1)[1].strip()
                    self._write_arduino(f"STEER:{angle_value}")

                # 2. System command
                else:
                    print(f'[PI] CMD <- {msg}')
                    cmd = msg.upper()
                    if cmd == 'START':
                        self._set_streaming(True)
                        self._write_arduino('START')
                    elif cmd == 'STOP':
                        self._set_streaming(False)
                        self._write_arduino('STOP')
                    else:
                        self._write_arduino(msg)
                        
            except Exception as e:
                print(f"[PI] Listener Error: {e}")

    def run(self):
        t_stream = threading.Thread(target=self._stream_loop, daemon=True)
        # Replaced separate cmd and status threads with one unified thread
        t_listener = threading.Thread(target=self._unified_listener, daemon=True)

        t_stream.start()
        t_listener.start()

        print('[PI] Ready. Waiting for START from laptop/app...')
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print('\n[PI] Shutting down...')
            self.running = False
        finally:
            try:
                self.front_cam.stop()
                self.rear_cam.stop()
                self.cmd_sock.close()
                self.tx_front.close()
                self.tx_rear.close()
            except Exception:
                pass
            if self.arduino:
                try:
                    self.arduino.close()
                except Exception:
                    pass


def main():
    bridge = PiBridge()
    bridge.run()


if __name__ == '__main__':
    main()