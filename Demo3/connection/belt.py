import socket
import time
import Demo3.states.globals as g
from Demo3.config.configs import *

# UDP Configuration
PORT = "8888"
IP = "192.168.118.195"

# Initialize socket immediately so sending works without "server" start
udp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    udp_server_socket.bind(("0.0.0.0", ESP32_STATUS_PORT))
    print(f"[ESP32 STATUS] UDP Bound to local port {ESP32_STATUS_PORT}")
except Exception as e:
    print(f"[ESP32 STATUS] Warning: Could not bind to specific port: {e}")


def run_esp32_status_server():
    """
    Previously handled client registration.
    Now just a placeholder to keep the thread alive if started by external logic.
    """
    print(f"[ESP32 STATUS] Ready. Sending direct UDP to {IP}:{PORT}")
    while True:
        time.sleep(60)


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
        # If both feet are out, prioritize the latest one
        if g.left_out_since > g.right_out_since:
            return "2"
        else:
            return "1"

    if left_ok:
        return "2" # 2 for left foot out
    if right_ok:
        return "1" # 1 for right foot out

    return "0"


def push_rear_status_to_esp32():
    with g.state_lock:
        rear = g.latest_state.get("rear", {})
        raw_rear_status = rear.get("status", "Inside")
        filtered_rear_status = _debounced_rear_status(raw_rear_status)

    # Only send payload if feet are out ("1" or "2").
    # If "0" (Inside), do not send anything.
    if filtered_rear_status == "0":
        return

    data = (filtered_rear_status + "\n").encode("utf-8")

    # Send directly to hardcoded IP/PORT
    try:
        udp_server_socket.sendto(data, (IP, int(PORT)))
    except Exception as e:
        print(f"Failed to send to {IP}:{PORT}: {e}")

def _run_belt_test_loop():
    """
    Standalone test loop for belt UDP link.
    - Periodically updates mock rear status
    - Pushes payload to target IP
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

    # Expand states to persist long enough for debounce (1.0s)
    # 0.2s * 10 = 2.0s per state
    expanded_states = []
    for s in test_states:
        expanded_states.extend([s] * 10)

    idx = 0
    print("[BELT TEST] Starting standalone test loop. Press Ctrl+C to stop.")
    print(f"[BELT TEST] Targeting {IP}:{PORT}...")

    try:
        while True:
            # Feed mock rear status so debounced logic can be observed.
            current_state = expanded_states[idx % len(expanded_states)]
            with g.state_lock:
                g.latest_state.setdefault("rear", {})
                g.latest_state["rear"]["status"] = current_state

            # Send to all connected ESP32 clients.
            push_rear_status_to_esp32()

            # Optional local debug print
            if idx % 10 == 0:
                 print(f"[BELT TEST] raw rear status = {current_state}")

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