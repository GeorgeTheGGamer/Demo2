"""Test connection functionalities."""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import threading
import time

import cv2
from Demo3.connection.commands import run_tcp_server, receive_rear_video, receive_front_video
from Demo3.vision.frame_processor import make_placeholder_frame
import Demo3.states.globals as g
from Demo3.config.configs import *


def start_pi_connections():
    threading.Thread(target=run_tcp_server, daemon=True).start()
    threading.Thread(target=receive_rear_video, daemon=True).start()
    threading.Thread(target=receive_front_video, daemon=True).start()


def main():
    print("[TEST] Starting Pi connection threads...")
    start_pi_connections()
    front_placeholder = make_placeholder_frame("Front AI Camera")
    rear_placeholder = make_placeholder_frame("Rear Backup Camera")
    print("[TEST] Displaying Pi video streams. Press 'q' to quit.")

    while True:
        now_ts = time.time()

        front = (
        g.latest_front_frame if g.front_frame_ts >0 and (now_ts - g.front_frame_ts) <= FRAME_MAX_AGE_SEC else None )
        rear = (
        g.latest_rear_frame if g.rear_frame_ts >0 and (now_ts - g.rear_frame_ts) <= FRAME_MAX_AGE_SEC else None )

        front_display = front_placeholder.copy() if front is None else front.copy()
        rear_display = rear_placeholder.copy() if rear is None else rear.copy()

        cv2.putText(front_display, "MODE: STREAM TEST", (20,35), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0,255,255),2)
        cv2.putText(rear_display, "MODE: STREAM TEST", (20,35), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0,255,255),2)

        cv2.imshow("Front AI Camera", front_display)
        cv2.imshow("Rear Backup Camera", rear_display)

        if cv2.waitKey(1) &0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
 main()
