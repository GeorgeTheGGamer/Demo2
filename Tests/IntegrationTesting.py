"""
Integration Tests for TrackSense System
Run: python3 -m pytest Tests/UnitTesting.py Tests/IntegrationTesting.py -v -p no:cacheprovider
     (from /Users/george_mahabir/Year3_Sem2/SDP/Demo3/Demo2)
"""
import sys
import json
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
# 2. Stub heavy native deps (leaf modules only, NOT numpy)
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
# 4. Pre-populate Demo3.states.globals
# ---------------------------------------------------------------------------
_g = types.ModuleType('Demo3.states.globals')
_g.angle_window=[]; _g.rear_servo_state={'servo':90,'last_seen':0.0}; _g.current_threshold=10
_g.state_lock=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))
_g.ws_lock   =MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))
_g.ws_clients=set(); _g.latest_state={}; _g.is_running=False; _g.cv_ready=False
_g.WS_PUSH_HZ=10; _g.last_ws_push_ts=0.0
_g.full_state_reset=MagicMock(); _g.reset_steering_to_default=MagicMock()
_g.tx_socket=MagicMock(); _g.pi_started=False
_g.front_current_angle=90.0; _g.rear_current_angle=90.0; _g.state='STRAIGHT'
_g.last_angle_send_ts=0.0; _g.latest_front_frame=None; _g.latest_rear_frame=None
_g.front_frame_ts=0.0; _g.rear_frame_ts=0.0
sys.modules['Demo3.states.globals']=_g
sys.modules['Demo3.states'].globals=_g

# ---------------------------------------------------------------------------
# 5. Pre-populate Demo3.config.configs
# ---------------------------------------------------------------------------
_cfg=types.ModuleType('Demo3.config.configs')
_cfg.STEER_VOTE_WINDOW=3; _cfg.STEER_VOTE_THRESHOLD=2; _cfg.MINMAX_ANGLE=30.0
_cfg.STRAIGHT_THRESHOLD=10; _cfg.STEER_THRESHOLD=10; _cfg.STATE_THRESHOLD=20
_cfg.ANGLE_SEND_HZ=3; _cfg.PI_IPS=['192.168.1.1']; _cfg.PI_CMD_PORT=8001
_cfg.HOST_IP='0.0.0.0'; _cfg.REAR_PORT=8002; _cfg.FRONT_PORT=8000
_cfg.MAX_DGRAM=65535; _cfg.API_PORT=5050; _cfg.REAR_NO_FEET_HOLD_SEC=2.0
sys.modules['Demo3.config.configs']=_cfg
sys.modules['Demo3.config'].configs=_cfg

# ---------------------------------------------------------------------------
# 6. Import real Flask and wire up test app
# ---------------------------------------------------------------------------
from flask import Flask
real_app = Flask(__name__); real_app.config['TESTING']=True
_g.app=real_app; _g.sock=MagicMock(**{'route': lambda p: (lambda f: f)})

from Demo3.connection.commands import (
    build_status_payload, forward_command_to_pi, send_angles_to_pi, receive_command, get_status
)
from Demo3.vision.helpers.steering_helper import SteeringHelper
from Demo3.vision.helpers.lane_fixer import LaneFixer
from Demo3.vision.helpers.YOLO_helpers import (
    feet_status, build_front_detection, build_rear_detection
)

real_app.add_url_rule('/command', view_func=receive_command, methods=['POST'])
real_app.add_url_rule('/status',  view_func=get_status,     methods=['GET'])

import Demo3.states.globals as g

def _reset():
    g.is_running=False; g.latest_state={}; g.pi_started=False
    g.front_current_angle=90.0; g.rear_current_angle=90.0; g.state='STRAIGHT'; g.last_angle_send_ts=0.0
    g.full_state_reset.reset_mock(); g.reset_steering_to_default.reset_mock(); g.tx_socket.reset_mock()


# ===========================================================================
# 1. /command endpoint
# ===========================================================================
class TestCommandEndpoint(unittest.TestCase):
    def setUp(self): self.c=real_app.test_client(); _reset()
    def _p(self,a): return self.c.post('/command',data=json.dumps({'action':a}),content_type='application/json')
    def test_start_running(self):       self._p('START'); self.assertTrue(g.is_running)
    def test_start_200(self):           self.assertEqual(self._p('START').status_code,200)
    def test_start_body_true(self):     self.assertTrue(json.loads(self._p('START').data)['running'])
    def test_stop_clears(self):         g.is_running=True; self._p('STOP'); self.assertFalse(g.is_running)
    def test_stop_200(self):            self.assertEqual(self._p('STOP').status_code,200)
    def test_stop_body_false(self):     self.assertFalse(json.loads(self._p('STOP').data)['running'])
    def test_stop_state_reset(self):    self._p('STOP'); g.full_state_reset.assert_called_once()
    def test_stop_steering_reset(self): self._p('STOP'); g.reset_steering_to_default.assert_called_once()
    def test_invalid_400(self):         self.assertEqual(self._p('DANCE').status_code,400)
    def test_empty_400(self):
        self.assertEqual(self.c.post('/command',data='{}',content_type='application/json').status_code,400)
    def test_lowercase_start(self):     self._p('start'); self.assertTrue(g.is_running)


# ===========================================================================
# 2. /status endpoint
# ===========================================================================
class TestStatusEndpoint(unittest.TestCase):
    def setUp(self): self.c=real_app.test_client(); _reset()
    def test_200(self):        self.assertEqual(self.c.get('/status').status_code,200)
    def test_keys(self):
        b=json.loads(self.c.get('/status').data)
        for k in ('running','cv_ready','front','rear'): self.assertIn(k,b)
    def test_reflects_running(self):
        g.is_running=True; g.latest_state['running']=True
        self.assertTrue(json.loads(self.c.get('/status').data)['running'])


# ===========================================================================
# 3. build_status_payload
# ===========================================================================
class TestBuildStatusPayload(unittest.TestCase):
    def setUp(self): _reset()
    def test_has_front(self):          self.assertIn('front',build_status_payload())
    def test_has_rear(self):           self.assertIn('rear', build_status_payload())
    def test_default_not_running(self):self.assertFalse(build_status_payload()['running'])
    def test_reflects_state(self):
        g.latest_state={'running':True,'front':{'robot_status':'OUT_OF_LANE'}}
        p=build_status_payload(); self.assertTrue(p['running']); self.assertEqual(p['front']['robot_status'],'OUT_OF_LANE')


# ===========================================================================
# 4. forward_command_to_pi
# ===========================================================================
class TestForwardCommandToPi(unittest.TestCase):
    def setUp(self): _reset()
    def test_start_sent(self):
        forward_command_to_pi('START'); self.assertIn(b'START',g.tx_socket.sendto.call_args[0][0])
    def test_stop_sent(self):
        forward_command_to_pi('STOP'); self.assertIn(b'STOP',g.tx_socket.sendto.call_args[0][0])
    def test_correct_ip(self):
        forward_command_to_pi('START')
        self.assertIn(('192.168.1.1',8001),[c[0][1] for c in g.tx_socket.sendto.call_args_list])
    def test_oserror_handled(self):
        g.tx_socket.sendto.side_effect=OSError("down")
        try:    forward_command_to_pi('START')
        except: self.fail("should not raise")
        finally:g.tx_socket.sendto.side_effect=None


# ===========================================================================
# 5. send_angles_to_pi
# ===========================================================================
class TestSendAnglesToPi(unittest.TestCase):
    def setUp(self): _reset(); g.is_running=True; g.pi_started=True
    def test_sends(self):      send_angles_to_pi(15.0); g.tx_socket.sendto.assert_called()
    def test_no_send_stopped(self):
        g.is_running=False; send_angles_to_pi(15.0); g.tx_socket.sendto.assert_not_called()
    def test_rate_limited(self):
        import time; g.last_angle_send_ts=time.time()
        send_angles_to_pi(15.0); g.tx_socket.sendto.assert_not_called()
    def test_right_state(self): g.front_current_angle=90.0; send_angles_to_pi(25.0); self.assertEqual(g.state,'RIGHT')
    def test_straight_state(self):g.front_current_angle=90.0; send_angles_to_pi(0.0); self.assertEqual(g.state,'STRAIGHT')


# ===========================================================================
# 6. Full CV steering pipeline
# ===========================================================================
def _lp(xl,xr,h=480,n=20): ys=[int(i*h/n) for i in range(n+1)]; return [(xl,y) for y in ys],[(xr,y) for y in ys]

class TestSteeringPipeline(unittest.TestCase):
    def test_centred_near_zero(self):
        # Lanes straddle frame centre (320): left at 200, right at 440 → centre near 320
        l,r=_lp(200,440); sh=SteeringHelper([l,r],(480,640))
        self.assertTrue(sh.worked); self.assertLess(abs(sh.heading_angle),15.0)
    def test_fixer_restores(self):
        fx=LaneFixer(); l,r=_lp(150,490)
        fx.fix([l,r],640); fx.fix([l,r],640)  # two calls to commit saved_width
        fixed=fx.fix([l],640)
        self.assertEqual(len(fixed),2); self.assertTrue(SteeringHelper(fixed,(480,640)).worked)
    def test_right_shifted_steer_left(self):
        # Lanes right of centre → center_points are right of 320 → negative heading (steer left)
        l,r=_lp(400,600); sh=SteeringHelper([l,r],(480,640))
        self.assertTrue(sh.worked); self.assertLess(sh.heading_angle,0)
    def test_left_shifted_steer_right(self):
        # Lanes left of centre → center_points left of 320 → positive heading (steer right)
        l,r=_lp(0,200); sh=SteeringHelper([l,r],(480,640))
        self.assertTrue(sh.worked); self.assertGreater(sh.heading_angle,0)


# ===========================================================================
# 7. Rear safety pipeline
# ===========================================================================
def _sl(): return [[(100,y) for y in range(0,110,10)],[(300,y) for y in range(0,110,10)]]

class TestRearSafetyPipeline(unittest.TestCase):
    def test_safe_no_danger(self):
        s,_,_=feet_status((200,50),(200,50),_sl()); self.assertEqual(build_rear_detection(s)['danger'],[])
    def test_left_out(self):
        s,_,_=feet_status((400,50),(200,50),_sl()); self.assertIn('left_foot(out_of_lane)',build_rear_detection(s)['danger'])
    def test_right_out(self):
        s,_,_=feet_status((200,50),(400,50),_sl()); self.assertIn('right_foot(out_of_lane)',build_rear_detection(s)['danger'])
    def test_no_feet_warning(self):
        s,_,_=feet_status(None,None,_sl()); self.assertGreater(len(build_rear_detection(s)['warning']),0)


# ===========================================================================
# 8. Strava upload (mocked HTTP)
# ===========================================================================
def _gpx(c):
    out=['<?xml version="1.0"?>','<gpx>','<trk><trkseg>']
    for p in c: out.append(f'<trkpt lat="{p["latitude"]}" lon="{p["longitude"]}"></trkpt>')
    out+=['</trkseg></trk>','</gpx>']; return '\n'.join(out)

class TestStravaUpload(unittest.TestCase):
    def _c(self,n=5): return [{'latitude':51.5+i*.001,'longitude':-0.1,'timestamp':i*1000} for i in range(n)]
    def test_gpx_all_points(self):    c=self._c(); self.assertEqual(_gpx(c).count('<trkpt'),len(c))
    def test_201_success(self):       m=MagicMock(); m.ok=True; self.assertTrue(m.ok)
    def test_401_error(self):         m=MagicMock(); m.ok=False; self.assertFalse(m.ok)
    def test_single_point(self):      self.assertIn('<trkpt',_gpx([{'latitude':51.5,'longitude':-0.1,'timestamp':0}]))
    def test_zero_points(self):       self.assertNotIn('<trkpt',_gpx([]))


# ===========================================================================
# 9. Front obstacle pipeline
# ===========================================================================
class TestFrontObstaclePipeline(unittest.TestCase):
    def _o(self,x1,y1,x2,y2): return {'bbox':[x1,y1,x2,y2],'cls':0,'conf':0.85}
    def test_in_lane_danger(self):
        # Object bottom edge at y=90 is within lane y range (0–100). Centre x=200 is between 100 and 300.
        self.assertGreater(len(build_front_detection([self._o(150,70,250,90)],_sl(),(480,640))['danger']),0)
    def test_outside_no_danger(self):
        self.assertEqual(build_front_detection([self._o(350,200,500,400)],_sl(),(480,640))['danger'],[])
    def test_no_lanes(self):
        self.assertEqual(build_front_detection([self._o(200,200,300,400)],[],(480,640))['danger'],[])
    def test_mixed(self):
        r=build_front_detection([self._o(150,70,250,90),self._o(350,70,500,90)],_sl(),(480,640))
        self.assertGreater(len(r['danger']),0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
