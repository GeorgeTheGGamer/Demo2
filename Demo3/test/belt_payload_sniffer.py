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

    print(f"[BELT SNIFFER] Preparing UDP listener targeting {host}:{port} ...")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", 0)) # Bind to ephemeral port
    s.settimeout(2.0)

    print("[BELT SNIFFER] UDP socket bound. Sending registration packet...")

    # We must send a packet to the server so it registers us as a client
    try:
        s.sendto(b"SNIFFER_REGISTER", (host, port))
        print("[BELT SNIFFER] Registration sent. Waiting for payloads... (Ctrl+C to stop)")
    except Exception as e:
        print(f"[BELT SNIFFER] Failed to send registration: {e}")
        return

    try:
        while True:
            try:
                data, addr = s.recvfrom(4096)
                line = data.strip()
                if line:
                    print(f"[BELT SNIFFER] {line.decode('utf-8', errors='replace')}")
            except socket.timeout:
                # Periodically re-register / keep-alive
                s.sendto(b"SNIFFER_REGISTER", (host, port))
                continue
    except KeyboardInterrupt:
        print("\n[BELT SNIFFER] Stopped by user.")
    finally:
        s.close()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
