import json
import socket
import time
import Demo3.states.globals as g
from Demo3.config.configs import *


class _Esp32Client:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr


def _safe_close(conn):
    try:
        conn.close()
    except Exception:
        pass


def _remove_client(client):
    with g.esp32_clients_lock:
        if client in g.esp32_clients:
            g.esp32_clients.remove(client)
    _safe_close(client.conn)
    print(f"[ESP32 STATUS] Client removed: {client.addr}")


def run_esp32_status_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", ESP32_STATUS_PORT))
    server.listen(5)
    print(f"[ESP32 STATUS] Listening on 0.0.0.0:{ESP32_STATUS_PORT}")

    while True:
        conn, addr = server.accept()

        client = _Esp32Client(conn, addr)
        with g.esp32_clients_lock:
            g.esp32_clients.append(client)

        print(f"[ESP32 STATUS] Client connected: {addr}")


def _debounced_rear_status(raw_status: str) -> str:
    """Return filtered rear status: only report foot-out after continuous gate time."""
    now = time.time()

    # Track how long each side stays out
    if raw_status in ("Left out", "Both out"):
        if g.left_out_since is None:
            g.left_out_since = now
    else:
        g.left_out_since = None

    if raw_status in ("Right out", "Both out"):
        if g.right_out_since is None:
            g.right_out_since = now
    else:
        g.right_out_since = None

    left_ok = g.left_out_since is not None and (now - g.left_out_since) >= FOOT_OUT_WARN_DELAY_SEC
    right_ok = g.right_out_since is not None and (now - g.right_out_since) >= FOOT_OUT_WARN_DELAY_SEC

    if left_ok and right_ok:
        return "Both out"
    if left_ok:
        return "Left out"
    if right_ok:
        return "Right out"

    if raw_status in ("Left out", "Right out", "Both out"):
        return "Inside"

    return raw_status


def push_rear_status_to_esp32():
    with g.state_lock:
        rear = g.latest_state.get("rear", {})
        raw_rear_status = rear.get("status", "Inside")
        filtered_rear_status = _debounced_rear_status(raw_rear_status)

        payload = {
            "rear_status": filtered_rear_status,
            "lane_status": g.state
        }

    data = (json.dumps(payload) + "\n").encode("utf-8")
    dead = []
    with g.esp32_clients_lock:
        clients_snapshot = list(g.esp32_clients)

    for client in clients_snapshot:
        try:
            client.conn.sendall(data)
        except Exception:
            dead.append(client)

    for client in dead:
        _remove_client(client)

def _run_belt_test_loop():
    """
    Standalone test loop for belt TCP link.
    - Starts ESP32 status server
    - Periodically updates mock rear status
    - Pushes payload to connected clients
    """
    test_states = [
        "Inside",
        "Left out",
        "Left out",
        "Left out",
        "Inside",
        "Right out",
        "Right out",
        "Both out",
        "Both out",
        "No feet detected",
    ]

    idx = 0
    print("[BELT TEST] Starting standalone test loop. Press Ctrl+C to stop.")
    print("[BELT TEST] Waiting ESP32 client to connect...")

    try:
        while True:
            # Feed mock rear status so debounced logic can be observed.
            with g.state_lock:
                g.latest_state.setdefault("rear", {})
                g.latest_state["rear"]["status"] = test_states[idx % len(test_states)]
                g.state = "STRAIGHT"

            # Send to all connected ESP32 clients.
            push_rear_status_to_esp32()

            # Optional local debug print
            raw_status = test_states[idx % len(test_states)]
            print(f"[BELT TEST] raw rear status = {raw_status}")

            idx += 1
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[BELT TEST] Stopped by user.")


if __name__ == "__main__":
    # Standalone mode: run only belt server + push loop
    # Does not require starting full laptop_server.py
    import threading

    threading.Thread(target=run_esp32_status_server, daemon=True).start()
    _run_belt_test_loop()