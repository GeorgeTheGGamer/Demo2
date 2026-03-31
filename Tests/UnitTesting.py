"""
Unit Tests for TrackSense System
Run: python3 -m pytest Tests/UnitTesting.py Tests/IntegrationTesting.py -v -p no:cacheprovider
     (from /Users/george_mahabir/Year3_Sem2/SDP/Demo3/Demo2)
"""
import sys
import math
import types
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Project root on path
# ---------------------------------------------------------------------------
_ROOT = '/Users/george_mahabir/Year3_Sem2/SDP/Demo3/Demo2'
_D3   = _ROOT + '/Demo3'
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# 2. Stub heavy native deps — leaf modules only, NOT numpy
# ---------------------------------------------------------------------------
def _stub(name, attrs=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]

_stub('torch', {'inference_mode': lambda: (lambda f: f), 'no_grad': lambda: (lambda f: f)})
_stub('torch.nn'); _stub('torch.nn.functional')
_stub('cv2'); _stub('ultralytics'); _stub('flask_sock')

# ---------------------------------------------------------------------------
# 3. Register Demo3 parent packages as proper namespace packages with __path__
#    This allows `import Demo3.states.globals as g` to traverse the tree.
# ---------------------------------------------------------------------------
def _pkg(name, path):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        mod.__package__ = name
        sys.modules[name] = mod
    return sys.modules[name]

_pkg('Demo3',            _D3)
_pkg('Demo3.states',     _D3 + '/states')
_pkg('Demo3.vision',     _D3 + '/vision')
_pkg('Demo3.vision.helpers', _D3 + '/vision/helpers')
_pkg('Demo3.connection', _D3 + '/connection')
_pkg('Demo3.config',     _D3 + '/config')

# ---------------------------------------------------------------------------
# 4. Pre-populate Demo3.states.globals before steering_helper imports it
# ---------------------------------------------------------------------------
_g = types.ModuleType('Demo3.states.globals')
_g.angle_window = []; _g.rear_servo_state = {'servo': 90, 'last_seen': 0.0}
_g.current_threshold = 10
_g.state_lock = MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))
_g.ws_lock    = MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))
_g.ws_clients = set(); _g.latest_state = {}; _g.is_running = False; _g.cv_ready = False
_g.WS_PUSH_HZ = 10; _g.last_ws_push_ts = 0.0; _g.app = MagicMock(); _g.sock = MagicMock()
_g.full_state_reset = MagicMock(); _g.reset_steering_to_default = MagicMock()
_g.tx_socket = MagicMock(); _g.pi_started = False
_g.front_current_angle = 90.0; _g.rear_current_angle = 90.0; _g.state = 'STRAIGHT'
_g.last_angle_send_ts = 0.0; _g.latest_front_frame = None; _g.latest_rear_frame = None
_g.front_frame_ts = 0.0; _g.rear_frame_ts = 0.0
sys.modules['Demo3.states.globals'] = _g
# Make it accessible as an attribute of its parent package
sys.modules['Demo3.states'].globals = _g

# ---------------------------------------------------------------------------
# 5. Pre-populate Demo3.config.configs with wildcard attrs
# ---------------------------------------------------------------------------
_cfg = types.ModuleType('Demo3.config.configs')
_cfg.STEER_VOTE_WINDOW=3; _cfg.STEER_VOTE_THRESHOLD=2; _cfg.MINMAX_ANGLE=30.0
_cfg.STRAIGHT_THRESHOLD=10; _cfg.STEER_THRESHOLD=10; _cfg.STATE_THRESHOLD=20
_cfg.ANGLE_SEND_HZ=3; _cfg.PI_IPS=['192.168.1.1']; _cfg.PI_CMD_PORT=8001
_cfg.HOST_IP='0.0.0.0'; _cfg.REAR_PORT=8002; _cfg.FRONT_PORT=8000
_cfg.MAX_DGRAM=65535; _cfg.API_PORT=5050; _cfg.REAR_NO_FEET_HOLD_SEC=2.0
sys.modules['Demo3.config.configs'] = _cfg
sys.modules['Demo3.config'].configs = _cfg

# ---------------------------------------------------------------------------
# 6. Real module imports
# ---------------------------------------------------------------------------
from Demo3.vision.helpers.lane_helpers import (
    interpolate_x_at_y, is_point_in_lane, normalize_point
)
from Demo3.vision.helpers.steering_helper import (
    SteeringHelper, rear_angle_to_servo, calculate_angle_to_center
)
from Demo3.vision.helpers.lane_fixer import LaneFixer
from Demo3.vision.helpers.focus_helper import FocusHelper
from Demo3.vision.helpers.YOLO_helpers import (
    feet_status, calculate_midpoint, build_front_detection, build_rear_detection
)


# ===========================================================================
# 1. interpolate_x_at_y
# ===========================================================================
class TestInterpolateXAtY(unittest.TestCase):
    def test_midpoint(self):         self.assertAlmostEqual(interpolate_x_at_y([(0,0),(100,10)], 5), 50.0)
    def test_exact_start(self):      self.assertAlmostEqual(interpolate_x_at_y([(0,0),(100,100)], 0), 0.0)
    def test_exact_end(self):        self.assertAlmostEqual(interpolate_x_at_y([(0,0),(100,100)], 100), 100.0)
    def test_out_of_range_none(self): self.assertIsNone(interpolate_x_at_y([(0,0),(100,50)], 200))
    def test_single_point_none(self): self.assertIsNone(interpolate_x_at_y([(50,50)], 50))


# ===========================================================================
# 2. is_point_in_lane
# ===========================================================================
class TestIsPointInLane(unittest.TestCase):
    def _lanes(self):
        return [[(100,y) for y in range(0,110,10)], [(300,y) for y in range(0,110,10)]]
    def test_inside(self):         self.assertTrue(is_point_in_lane((200,50), self._lanes()))
    def test_outside(self):        self.assertFalse(is_point_in_lane((400,50), self._lanes()))
    def test_none_point(self):     self.assertFalse(is_point_in_lane(None, self._lanes()))
    def test_one_lane_only(self):  self.assertFalse(is_point_in_lane((150,50), [self._lanes()[0]]))
    def test_on_boundary(self):    self.assertTrue(is_point_in_lane((100,50), self._lanes()))


# ===========================================================================
# 3. normalize_point
# ===========================================================================
class TestNormalizePoint(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(normalize_point(None))
    def test_zero_zero_returns_none(self):
        self.assertIsNone(normalize_point(MagicMock(**{'__getitem__.side_effect': lambda i: 0.0})))
    def test_valid_point(self):
        self.assertEqual(normalize_point(MagicMock(**{'__getitem__.side_effect': lambda i: [12.0,34.0][i]})), (12.0, 34.0))


# ===========================================================================
# 4. SteeringHelper
# ===========================================================================
def _lp(xl=200, xr=400, h=480, n=10):
    ys=[int(i*h/n) for i in range(n+1)]
    return [(xl,y) for y in ys], [(xr,y) for y in ys]

class TestSteeringHelper(unittest.TestCase):
    def test_worked_true(self):      self.assertTrue(SteeringHelper([*_lp()],(480,640)).worked)
    def test_empty_not_worked(self): self.assertFalse(SteeringHelper([],(480,640)).worked)
    def test_centerline_between_lanes(self):
        l,r=_lp(100,300)
        for cx,_ in SteeringHelper([l,r],(480,640)).center_points:
            self.assertGreater(cx,90); self.assertLess(cx,310)
    def test_straight_near_zero(self):
        # Lanes straddle the centre (320) so center_points are near x=320, giving ~zero heading
        l,r=_lp(200,440); self.assertAlmostEqual(SteeringHelper([l,r],(480,640)).heading_angle, 0.0, places=0)
    def test_single_lane_fallback(self):
        l,_=_lp(); self.assertTrue(SteeringHelper([l],(480,640)).worked)
    def test_too_few_points_not_worked(self):
        self.assertFalse(SteeringHelper([[(100,10),(110,20)]],(480,640)).worked)


# ===========================================================================
# 5. rear_angle_to_servo
# ===========================================================================
class TestRearAngleToServo(unittest.TestCase):
    def test_zero_is_90(self):        self.assertEqual(rear_angle_to_servo(0.0), 90)
    def test_positive_below_90(self): self.assertLess(rear_angle_to_servo(20.0), 90)
    def test_negative_above_90(self): self.assertGreater(rear_angle_to_servo(-20.0), 90)
    def test_clamp_high(self):        self.assertEqual(rear_angle_to_servo(999.0), 45)
    def test_clamp_low(self):         self.assertEqual(rear_angle_to_servo(-999.0), 135)
    def test_in_range(self):
        for a in range(-50,51):
            v=rear_angle_to_servo(float(a)); self.assertGreaterEqual(v,45); self.assertLessEqual(v,135)
    def test_snapped_5deg(self):
        for a in range(-45,46,5): self.assertEqual(rear_angle_to_servo(float(a))%5, 0)


# ===========================================================================
# 6. calculate_angle_to_center
# ===========================================================================
class TestCalculateAngleToCenter(unittest.TestCase):
    def _f(self): f=MagicMock(); f.shape=(480,640,3); return f
    def test_centre_zero(self):    self.assertAlmostEqual(calculate_angle_to_center((320,240),self._f()), 0.0, places=4)
    def test_none_none(self):      self.assertIsNone(calculate_angle_to_center(None,self._f()))
    def test_right_positive(self): self.assertGreater(calculate_angle_to_center((480,240),self._f()), 0)
    def test_left_negative(self):  self.assertLess(calculate_angle_to_center((160,240),self._f()), 0)


# ===========================================================================
# 7. LaneFixer
# ===========================================================================
def _col(x, n=10): return [(x,int(i*100/n)) for i in range(n+1)]

class TestLaneFixer(unittest.TestCase):
    def setUp(self): self.f=LaneFixer()
    def test_both_unchanged(self):  self.assertEqual(len(self.f.fix([_col(100),_col(300)],640)), 2)
    def test_right_synthesised(self):
        # Train with two consecutive observations so saved_width is set
        self.f.fix([_col(100),_col(300)],640)
        self.f.fix([_col(100),_col(300)],640)  # second call commits saved_width
        self.assertEqual(len(self.f.fix([_col(100)],640)), 2)
    def test_left_synthesised(self):
        self.f.fix([_col(100),_col(300)],640)
        self.f.fix([_col(100),_col(300)],640)  # second call commits saved_width
        self.assertEqual(len(self.f.fix([_col(300)],640)), 2)
    def test_empty_unchanged(self): self.assertEqual(self.f.fix([],640), [])
    def test_width_learnt(self):
        self.f.fix([_col(100),_col(300)],640)
        self.f.fix([_col(100),_col(300)],640)  # second call commits saved_width
        self.assertAlmostEqual(self.f.saved_width, 200.0, delta=5)
    def test_outlier_rejected(self):
        self.f.fix([_col(100),_col(300)],640); w=self.f.saved_width; self.f.update(5000.0); self.assertEqual(self.f.saved_width, w)


# ===========================================================================
# 8. FocusHelper
# ===========================================================================
class TestFocusHelper(unittest.TestCase):
    def test_empty_none_pair(self): self.assertEqual(FocusHelper().focus([]), (None,None))
    def test_tracks_closest(self):
        fh=FocusHelper(640,480); fh.get_midpoint=lambda p:(p[0],p[1])
        fh.focus([(100,100)]); self.assertEqual(fh.focus([(105,105),(500,500)]), (105,105))


# ===========================================================================
# 9. feet_status & calculate_midpoint
# ===========================================================================
def _sln(): return [[(100,y) for y in range(0,110,10)],[(300,y) for y in range(0,110,10)]]

class TestFeetStatus(unittest.TestCase):
    def test_safe(self):     self.assertEqual(feet_status((200,50),(200,50),_sln())[0],'Safe')
    def test_both_out(self): self.assertEqual(feet_status((400,50),(400,50),_sln())[0],'Both out')
    def test_no_feet(self):  self.assertEqual(feet_status(None,None,_sln())[0],'No feet detected')
    def test_left_out(self): self.assertEqual(feet_status((400,50),(200,50),_sln())[0],'Left out')
    def test_right_out(self):self.assertEqual(feet_status((200,50),(400,50),_sln())[0],'Right out')

class TestCalculateMidpoint(unittest.TestCase):
    def test_midpoint(self):   self.assertEqual(calculate_midpoint([0.,0.],[100.,100.]),[50.,50.])
    def test_none_left(self):  self.assertIsNone(calculate_midpoint(None,[1.,1.]))
    def test_none_right(self): self.assertIsNone(calculate_midpoint([1.,1.],None))


# ===========================================================================
# 10. build_front / build_rear detection
# ===========================================================================
class TestBuildFrontDetection(unittest.TestCase):
    def _o(self,x1,y1,x2,y2): return {'bbox':[x1,y1,x2,y2],'cls':0,'conf':0.9}
    def test_in_lane_danger(self):
        self.assertGreater(len(build_front_detection([self._o(150,20,250,50)],_sln(),(480,640))['danger']),0)
    def test_out_no_danger(self):
        self.assertEqual(build_front_detection([self._o(400,20,500,50)],_sln(),(480,640))['danger'],[])
    def test_empty_objs(self):
        r=build_front_detection([],_sln(),(480,640)); self.assertEqual(r['danger'],[]); self.assertEqual(r['warning'],[])

class TestBuildRearDetection(unittest.TestCase):
    def test_safe_empty(self):     self.assertEqual(build_rear_detection('Safe')['danger'],[])
    def test_left_out(self):       self.assertIn('left_foot(out_of_lane)',build_rear_detection('Left out')['danger'])
    def test_right_out(self):      self.assertIn('right_foot(out_of_lane)',build_rear_detection('Right out')['danger'])
    def test_both_out_two(self):   self.assertEqual(len(build_rear_detection('Both out')['danger']),2)
    def test_no_feet_warning(self):self.assertGreater(len(build_rear_detection('No feet detected')['warning']),0)


# ===========================================================================
# 11. Haversine (Python port of useLocationTracking.js getDistance)
# ===========================================================================
def haversine(la1,lo1,la2,lo2):
    R=6371e3; p1,p2=math.radians(la1),math.radians(la2)
    a=math.sin(math.radians(la2-la1)/2)**2+math.cos(p1)*math.cos(p2)*math.sin(math.radians(lo2-lo1)/2)**2
    return 2*R*math.atan2(math.sqrt(a),math.sqrt(1-a))

class TestHaversine(unittest.TestCase):
    def test_same_zero(self):      self.assertAlmostEqual(haversine(51.5,-0.1,51.5,-0.1),0.0)
    def test_london_paris(self):   self.assertAlmostEqual(haversine(51.5074,-0.1278,48.8566,2.3522)/1000,341,delta=5)
    def test_symmetric(self):      self.assertAlmostEqual(haversine(10.,20.,10.5,20.5),haversine(10.5,20.5,10.,20.),places=5)
    def test_small_move_gt0(self): self.assertGreater(haversine(51.5,-0.1,51.5001,-0.1),0)


# ===========================================================================
# 12. GPX generation (port of generateGPXString in strava.js)
# ===========================================================================
def gpx(coords):
    out=['<?xml version="1.0"?>','<gpx>','<trk><trkseg>']
    for p in coords:
        out.append(f'<trkpt lat="{p["latitude"]}" lon="{p["longitude"]}"><time>2024-01-01T00:00:00Z</time></trkpt>')
    out+=['</trkseg></trk>','</gpx>']; return '\n'.join(out)

class TestGPXGeneration(unittest.TestCase):
    def test_structure(self):
        g=gpx([{'latitude':51.5,'longitude':-0.1,'timestamp':0}])
        self.assertIn('<gpx',g); self.assertIn('</gpx>',g); self.assertIn('<trkpt',g)
    def test_values_present(self):
        g=gpx([{'latitude':12.3,'longitude':-78.9,'timestamp':0}])
        self.assertIn('12.3',g); self.assertIn('-78.9',g)
    def test_empty_no_trkpt(self):    self.assertNotIn('<trkpt',gpx([]))
    def test_count(self):
        self.assertEqual(gpx([{'latitude':1.,'longitude':2.,'timestamp':0},{'latitude':3.,'longitude':4.,'timestamp':1}]).count('<trkpt'),2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
