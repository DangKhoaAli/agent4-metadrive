"""Offline tests for exact governor action handoff (MetaDrive not needed)."""
from __future__ import annotations
import benchmark_handoff_v362 as bench


def test_exact_reference_handoff():
    v36 = bench.v36
    v30 = v36.v30
    old = (v30.feature_for_model, v30.planned_geometry, v30.governed_speed_action,
           v36.make_row, v36.predict_action)
    features = {'speed_mps': 0.0}
    route = {'adapted': False, 'remaining_m': 100.0}
    geom = {'lateral_right_m': 0.0, 'heading_error_rad': 0.0}
    try:
        v30.feature_for_model = lambda env, max_gap: (features, route)
        v30.planned_geometry = lambda car, max_gap: geom
        v30.governed_speed_action = lambda speed, base, remaining: (0.19+speed/1000, base)
        v36.make_row = lambda *args: {}
        v36.predict_action = lambda bundle,row: 0.8
        saved = (v30.feature_for_model, v30.planned_geometry, v30.governed_speed_action)
        actions = []
        with bench.recorded_longitudinal('handoff_80', {}, [], actions):
            for i in range(85):
                v30.feature_for_model(None, 10)
                v30.planned_geometry(None, 10)
                applied, target = v30.governed_speed_action(i, 4.0, 100.0)
                if i<80:
                    assert abs(applied - (0.19+i/1000)) < 1e-12
                else:
                    assert applied == 0.8
                assert target == 4.0
        assert (v30.feature_for_model, v30.planned_geometry, v30.governed_speed_action)==saved
        assert len(actions)==85
        assert sum(a['reference_applied'] for a in actions)==80
        assert all(a['learned_prediction']==0.8 for a in actions)
        actions.clear()
        with bench.recorded_longitudinal('learned_long_v36', {}, [], actions):
            v30.feature_for_model(None,10)
            v30.planned_geometry(None,10)
            assert v30.governed_speed_action(0,4.0,100.0)[0]==0.8
        actions.clear()
        with bench.recorded_longitudinal('governor_v35', {}, [], actions):
            v30.feature_for_model(None,10)
            v30.planned_geometry(None,10)
            assert v30.governed_speed_action(0,4.0,100.0)[0]==0.19
        assert actions[0]['learned_prediction'] is None
    finally:
        (v30.feature_for_model,v30.planned_geometry,v30.governed_speed_action,
         v36.make_row,v36.predict_action)=old


def test_exception_restores():
    v30=bench.v36.v30
    old=(v30.feature_for_model,v30.planned_geometry,v30.governed_speed_action)
    try:
        with bench.recorded_longitudinal('handoff_160', {}, [], []):
            raise RuntimeError('deliberate')
    except RuntimeError:
        pass
    assert old==(v30.feature_for_model,v30.planned_geometry,v30.governed_speed_action)


def test_contract():
    assert bench.HANDOFF_STEPS['handoff_80']==80
    assert bench.HANDOFF_STEPS['handoff_160']==160
    assert bench.HANDOFF_STEPS['learned_long_v36']==0
    assert bench.HANDOFF_STEPS['governor_v35'] is None
    try:
        with bench.recorded_longitudinal('oops', {}, [], []):
            pass
    except ValueError:
        pass
    else: raise AssertionError('Bad profile not rejected')


def main():
    for t in (test_exact_reference_handoff, test_exception_restores, test_contract):
        t();print('PASS',t.__name__)
    print('V3.6.2 OFFLINE UNIT TESTS PASS; MetaDrive closed-loop not run.')

if __name__=='__main__':main()
