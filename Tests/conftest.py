"""conftest.py — shared pytest configuration for all TrackSense tests."""
import sys
import types

# Ensure project root is on path
sys.path.insert(0, '/Users/george_mahabir/Year3_Sem2/SDP/Demo3/Demo2')

# ---------------------------------------------------------------------------
# Pre-stub heavy native modules so tests run without GPU / ROS / hardware
# ---------------------------------------------------------------------------
def _add_stub(name, attrs=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]

_add_stub('torch', {'inference_mode': lambda: (lambda f: f), 'no_grad': lambda: (lambda f: f)})
_add_stub('torch.nn')
_add_stub('torch.nn.functional')
_add_stub('cv2')
_add_stub('ultralytics')

# stub flask_sock but NOT flask itself (we need the real Flask for integration tests)
_add_stub('flask_sock')
