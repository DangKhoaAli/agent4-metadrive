"""Pure math tests; run without MetaDrive or sklearn installed."""
import math
from residual_controller_v30 import (
    compute_correction, steering_action, estimated_edge_clearance_m,
    lateral_sign_from_lane_geometry, rightward_lateral_error)


class FakeLane:
    length = 100.0
    def position(self, s, lat):
        # Facing +x. Physical right points to -y.
        return (float(s), -float(lat))
    def local_coordinates(self, pos):
        return (float(pos[0]), -float(pos[1]))


class FakeCar:
    WIDTH = 1.852
    LENGTH = 4.515
    dist_to_left_side = 1.31
    dist_to_right_side = 9.19


def close(a,b,tol=1e-8):
    assert abs(a-b) < tol, f'{a} != {b}'


def main():
    lane = FakeLane()
    assert lateral_sign_from_lane_geometry(lane, 30) == 1
    offset, _, sign = rightward_lateral_error(lane, (30,-0.4))
    close(offset, 0.4)
    assert sign == 1
    # Right of lane -> left steering; vehicle rotated left -> right steering.
    assert compute_correction(0.5,0.0)['bounded'] > 0
    assert compute_correction(0.0,0.10)['bounded'] < 0
    b = steering_action(0.037753857359, 'baseline', lateral_right_m=-0.4, heading_error_rad=.14)
    close(b['steer'], -0.037753857359)
    r = steering_action(0.037753857359, 'residual', lateral_right_m=-0.4, heading_error_rad=.14)
    assert r['steer'] < b['steer'], (r,b)
    v = steering_action(-3.0,'residual',lateral_right_m=3,heading_error_rad=-1.0,steer_limit=0.35)
    close(v['steer'],0.35)
    z = steering_action(-3.0,'residual',lateral_right_m=3,heading_error_rad=-1.0,
                        steer_limit=0.35,previous=0.0,max_delta=0.12)
    close(z['steer'],0.12)
    edge = estimated_edge_clearance_m(FakeCar(), math.radians(8))
    assert 0.0 < edge['minimum'] < 0.2, edge
    # Reproduce a different lane-side convention without hardcoding signs.
    class ReversedLane(FakeLane):
        def position(self,s,lat): return (float(s),float(lat))
        def local_coordinates(self,pos): return (float(pos[0]),float(pos[1]))
    assert lateral_sign_from_lane_geometry(ReversedLane(),30) == -1
    close(rightward_lateral_error(ReversedLane(),(30,-0.4))[0], +0.4)
    print('Agent 4 V3.0 residual steering + geometry synthetic tests: PASS')

if __name__ == '__main__':
    main()
