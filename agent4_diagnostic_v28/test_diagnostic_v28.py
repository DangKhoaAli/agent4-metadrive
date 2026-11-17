"""No MetaDrive installation needed for these simple geometry/formatting checks."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_curve_v28 import angle_delta_radians, compact_lane_state, float_or_none, safe_flags

assert abs(angle_delta_radians(math.pi+0.1,0)+math.pi-0.1) < 1e-9
assert abs(angle_delta_radians(-math.pi-0.1,0)-math.pi+0.1) < 1e-9
assert float_or_none(float('nan')) is None
assert safe_flags({'out_of_road': True, 'route_completion': 0.33, 'not_in_scope': 42}) == {
    'out_of_road': True, 'route_completion': 0.33}
class FakeLane:
    length = 50.0
    width = 3.5
    def local_coordinates(self, pos):
        return pos[0], pos[1]
    def position(self, s, l):
        return s,l
class FakeAgent:
    lane = FakeLane()
    lane_index = ('a','b',0)
    heading_theta = 0.0
    position=(10,0.2)
r=compact_lane_state(FakeAgent())
assert abs(r['lane_lateral_m']-0.2)<1e-6
assert abs(r['heading_error_rad'])<1e-6
assert r['lane_longitudinal_in_range']
print('V2.8 diagnostic helpers: PASS')
