"""No-MetaDrive offline tests for V3.6.1 launch diagnostic patch."""
from __future__ import annotations
import math
import benchmark_launch_v361 as bench


def check_cap_and_restore():
    module=bench.v36
    real=module.predict_action
    try:
        module.predict_action=lambda bundle,row:row['v']
        fake=module.predict_action
        with bench.diagnostic_cap(3,.4) as s:
            vals=[module.predict_action(None, {'v':v}) for v in [1.,.7,-.5,.8,.6]]
            assert vals==[.4,.4,-.5,.8,.6], vals
            assert s['prediction_calls']==5
            assert s['cap_active_calls']==3
            assert s['capped_positive_calls']==2
            assert s['max_raw_launch_action']==1.
        assert module.predict_action is fake
        try:
            with bench.diagnostic_cap(2,.4):
                raise RuntimeError('check restore')
        except RuntimeError:
            pass
        assert module.predict_action is fake
    finally:
        module.predict_action=real


def check_inputs():
    for s,c in [(0,.4),(-1,.4),(201,.4),(4,0),(4,1),(4,float('nan'))]:
        try:bench.validate_launch_params(s,c)
        except ValueError:pass
        else:raise AssertionError((s,c))
    bench.validate_launch_params(40,.4)
    assert bench.PROFILES==('governor_v35','learned_long_v36','learned_launch_cap_v361')


def check_no_steering_or_plant_change():
    # Run_one uses V3.6.one; it may only patch prediction *during* learned launch cap.
    real=bench.v36.one
    original_predict=bench.v36.predict_action
    calls=[]
    def fake_one(env,map_name,seed,speed,profile,pdd,steering,long,steps,stride):
        calls.append(profile)
        if profile=='learned_long_v36':
            raw=[1.,.6,-.2,1.]
            applied=[bench.v36.predict_action(None,{'v':v}) for v in raw]
            assert applied==[.4,.4,-.2,1.]
        return {'map':map_name,'seed':seed,'speed_mps':speed,'profile':profile,
                'summary':{'n_applied_steps':4,'status':'arrive_dest','error_count':0},
                'errors':[], 'trace':[{'step':0,'speed_mps_before':0.,
                     'longitudinal_attribution_v36':{'mode':profile}}]}
    try:
        bench.v36.one=fake_one
        bench.v36.predict_action=lambda bundle,row:row['v']
        result=bench.run_one(None,'C',76,4.,'learned_launch_cap_v361',None,
                             None,None,1800,10,3,.4)
        assert result['profile']=='learned_launch_cap_v361'
        assert result['summary']['launch_diagnostic']['capped_positive_calls']==2
        assert result['summary']['error_count']==0
        assert result['trace'][0]['longitudinal_attribution_v36']['mode']=='learned_launch_cap_v361'
        assert bench.v36.predict_action(None,{'v':1.})==1.
    finally:
        bench.v36.one=real
        bench.v36.predict_action=original_predict


def main():
    for t in [check_cap_and_restore,check_inputs,check_no_steering_or_plant_change]:
        t();print('PASS',t.__name__)
    print('V3.6.1 OFFLINE TESTS PASS (does not validate live MetaDrive).')


if __name__=='__main__':main()
