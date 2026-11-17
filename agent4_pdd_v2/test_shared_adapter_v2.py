"""Fast self-tests that do not require MetaDrive installation or real PDD files."""
import math
from shared_adapter_v2 import FEATURES, pack_features, pdd_features, summarize_objects, cap_waypoint


def test_capping():
    x, y = cap_waypoint(100, 0)
    assert (x, y) == (50.0, 0.0)
    x, y = cap_waypoint(30, 40)
    assert abs(math.hypot(x, y) - 50) < 1e-9


def test_sectors():
    objects = [
        {'kind':'vehicle', 'x':20, 'y':0, 'speed_mps':8},
        {'kind':'vehicle', 'x':10, 'y':5, 'speed_mps':9},
        {'kind':'vehicle', 'x':3, 'y':-5, 'speed_mps':9},
        {'kind':'pedestrian', 'x':6, 'y':0, 'speed_mps':0},
    ]
    o = summarize_objects(objects, ego_speed_mps=10)
    assert o['has_front_vehicle'] == 1
    assert o['has_right_vehicle'] == 1
    assert o['has_left_vehicle'] == 1
    assert o['has_front_pedestrian'] == 1
    assert o['front_vehicle_closing_speed_mps'] == 2
    o2 = summarize_objects([], ego_speed_mps=0)
    assert o2['has_front_vehicle'] == 0 and o2['front_vehicle_dist_m'] == 60


def test_pdd_adapter():
    info = {'speed': 10, 'target_point': [50, 3], 'target_point_next': [100, 6],
            'lane_info': {'lane_width': 3.25}}
    boxes = [{'class':'ego_car','position':[0,0,0]},
             {'class':'car','position':[30,1,0], 'speed':5}]
    f = pdd_features(info, boxes)
    assert list(f) == FEATURES
    assert 0 < f['front_vehicle_dist_m'] < 60
    assert all(math.isfinite(v) for v in f.values())
    assert len(FEATURES) == 15


if __name__ == '__main__':
    test_capping()
    test_sectors()
    test_pdd_adapter()
    print('PASS: 3 adapter self-tests (PDD geometry, missing-object mask, feature schema).')
