import threading
import time
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Demo3.connection.belt import run_esp32_status_server


def main():
    print("[BELT TEST] Starting ESP32 status server...")
    # Pass function reference as target (do not call it here).
    threading.Thread(target=run_esp32_status_server, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[BELT TEST] Stopped by user.")


if __name__ == "__main__":
    main()