"""Storage for all configuration parameters"""

# --- PATH SETUP ---
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- CONFIGURATION ---
HOST_IP = "0.0.0.0"
API_PORT = 5050
FRONT_PORT = 8000
REAR_PORT = 8002
MAX_DGRAM = 65507

PI_IPS = ["192.168.118.199"]
PI_CMD_PORT = 8001

DEFAULT_CONFIG = 'configs/clrnet/clr_resnet18_tusimple.py'
DEFAULT_CHECKPOINT = 'checkpoints/tusimple_r18.pth'
DEFAULT_DEVICE = 'cpu' # Change to 'cuda' or 'cpu' as needed
DEFAULT_FRONT_YOLO = 'checkpoints/yolov8n_int8.tflite'
DEFAULT_REAR_POSE = 'checkpoints/yolov8n-pose_int8.tflite'
MINMAX_ANGLE = 30.0  # Max heading angle in degrees for servo mapping
STRAIGHT_THRESHOLD = 5 # angle within ± this value will be ignored
STEER_THRESHOLD = 10 # angle within ± this value will be ignored
STATE_THRESHOLD = 20 # If current_angle deviates from 90 by more than this, update state to LEFT/RIGHT

# --- STOP CONDITION HOLD TIMES (seconds) ---
# How long a condition must be continuously active before STOP is sent to Pi.
HOLD_OBJECT_IN_LANE_SEC    = 10.0   # Front: YOLO object inside lane
HOLD_CORNER_ANGLE_SEC      = 6.0   # Front: heading angle >= 70 degrees
HOLD_FRONT_NO_LANE_SEC     = 8.0   # Front: CLRNet detects 0 lanes
HOLD_FRONT_OUT_LANE_SEC    = 8.0   # Front: robot out of lane
HOLD_LEFT_FOOT_SEC         = 10.0   # Rear: left ankle outside lane
HOLD_RIGHT_FOOT_SEC        = 10.0   # Rear: right ankle outside lane
HOLD_BOTH_FEET_SEC         = 10.0   # Rear: both ankles outside lane
HOLD_NO_FEET_SEC           = 15.0   # Rear: no ankles detected at all
HOLD_REAR_NO_LANE_SEC      = 8.0   # Rear: CLRNet detects 0 lanes
HOLD_REAR_OUT_LANE_SEC     = 8.0   # Rear: robot out of lane

# TODO: Remove it? (not used)
# How long (seconds) the heading angle must remain within the same servo
# threshold band before the servo is actually updated. Prevents jitter on
# borderline angles and makes turning smoother.
STEER_HOLD_SEC = 3.0

# How long (seconds) with no feet detected before the rear servo resets to 90° (centre).
REAR_NO_FEET_HOLD_SEC = 5.0

# Max age (seconds) of a camera frame before it is treated as stale / no stream.
FRAME_MAX_AGE_SEC = 6.0

# Majority-vote steering: number of recent frames to consider and minimum
# agreement count required before a new servo value is committed.
# EMA smooths the raw angle first, then majority-vote confirms direction.
# E.g. window=5, threshold=3 means 3/5 EMA-smoothed frames must agree.
STEER_VOTE_WINDOW    = 5    # rolling window size (reduced for low-FPS responsiveness)
STEER_VOTE_THRESHOLD = 3    # minimum votes needed (out of STEER_VOTE_WINDOW)