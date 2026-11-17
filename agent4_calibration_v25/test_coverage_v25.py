"""Dependency and unit tests; no MetaDrive installation required."""
import math
from probe_coverage_v25 import (
    descriptive, wrapped_delta_rad, road_tokens, is_curve_lane,
    in_special_block, make_feature_variant, select_examples,
)


def test():
    assert descriptive([3., 1., 2.])['median'] == 2.
    assert abs(wrapped_delta_rad(-math.pi + 0.1, math.pi - 0.1) - 0.2) < 1e-9
    assert road_tokens(('1C0_0_', '2C0_0_', 0))[0] == ['1C0_0_', '2C0_0_']
    assert in_special_block(['1C0_0_', '2C0_0_'], 'C')
    assert in_special_block(['1X0_0_'], 'X')
    assert not in_special_block(['1S0_0_'], 'X')
    assert is_curve_lane('CircularLane') and not is_curve_lane('StraightLane')
    a = make_feature_variant({'speed_mps': 1}, [(25., -2.), (45., 1.)])
    assert a['wp1_forward_m'] == 25 and a['wp1_right_m'] == -2
    r = [{'seed': 0, 'step': i, 'geometry_error': None,
          'turn_motion': i == 3, 'in_curve_lane': False,
          'in_intersection_block': False} for i in range(5)]
    chosen = select_examples(r, 2)
    assert any(x['step'] == 3 for x in chosen)
    print('Agent4 V2.5 helper tests: PASS')


if __name__ == '__main__':
    test()
