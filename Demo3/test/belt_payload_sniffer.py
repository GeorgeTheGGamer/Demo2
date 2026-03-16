import socket
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Demo3.config.configs import ESP32_STATUS_PORT


def main():
    host = "127.0.0.1"
    port = ESP32_STATUS_PORT

    # Optional CLI: python -m Demo3.test.belt_payload_sniffer 127.0.0.1 5006
    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    print(f"[BELT SNIFFER] Connecting to {host}:{port} ...")
    s = socket.create_connection((host, port), timeout=5)
    s.settimeout(1.0)
    print("[BELT SNIFFER] Connected. Waiting for payloads... (Ctrl+C to stop)")

    buf = b""
    try:
        while True:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                continue

            if not chunk:
                print("[BELT SNIFFER] Connection closed by server.")
                break

            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                print(f"[BELT SNIFFER] {line.decode('utf-8', errors='replace')}")
    except KeyboardInterrupt:
        print("\n[BELT SNIFFER] Stopped by user.")
    finally:
        try:
            s.close()
        except Exception:
            pass
        time.sleep(0.05)


if __name__ == "__main__":
    main()
