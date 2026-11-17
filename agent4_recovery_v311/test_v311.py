"""Contract and synthetic action-kick tests. Not a MetaDrive driving test."""
from __future__ import annotations
import math
from recovery_v311 import KickSettings,DisturbanceMonitor,injected_step,parse_targets,aggregate,v310


def rejects(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected ValueError')


def test_settings():
    assert parse_targets('0.2,0.4')==[.2,.4]
    rejects(lambda:parse_targets('0.2,0.2'))
    rejects(lambda:parse_targets('0.8'))
    rejects(lambda:KickSettings(target_offset_m=.2,tolerance_m=.25).validate())
    assert KickSettings().validate().start_step==65

class Car:
    def __init__(self,drift=.06):
        self.offset=0.;self.heading=0.;self.speed=18.;self.speed_km_h=64.8;self.drift=drift

class Env:
    def __init__(self,drift=.06):
        self.agent=Car(drift);self.seen=[]
    def step(self,action):
        self.seen.append(list(action))
        if abs(action[0])>.05:
            self.agent.offset+=math.copysign(self.agent.drift,action[0]);self.agent.heading=0.015
        else:
            self.agent.offset*=.55;self.agent.heading*=.4
        return None,0.,False,False,{'route_completion':.1}


def exercise(sign=1,drift=.06):
    saved=v310.v30.planned_geometry
    v310.v30.planned_geometry=lambda agent,gap:{'lateral_right_m':agent.offset,'heading_error_rad':agent.heading}
    try:
        env=Env(drift);original=env.step
        s=KickSettings(start_step=5,target_offset_m=.2,sign=sign,steering_md=.1,
            max_kick_steps=10,stable_steps=3,window_steps=20,sample_stride=1)
        with injected_step(env,s) as monitor:
            for i in range(40):env.step([0.,.3])
        assert env.step==original.__func__.__get__(env,Env) or env.step.__func__==original.__func__
        return monitor,env
    finally:v310.v30.planned_geometry=saved


def test_success():
    for sign in (1,-1):
        mon,env=exercise(sign)
        result=mon.summary('arrive_dest')
        assert len(env.seen)==40
        assert math.isclose(mon.start_snapshot['speed_mps'],18.0,rel_tol=0,abs_tol=1e-9)
        assert result['target_reached'],result
        assert result['recovered'],result
        assert result['recovery_outcome']=='recovered',result
        assert result['kick_steps']>=3
        assert result['recovery_time_s']>=0
        assert abs(env.agent.offset)<.001
        assert result['measured_steps']==40
        assert result['measurement_errors']==[]
        assert any(t['injected_steer_md'] for t in mon.observations)
        assert result['peak_abs_offset_m']>=.2


def test_target_not_reached():
    m,_=exercise(1,drift=.001)
    result=m.summary('arrive_dest')
    assert not result['target_reached']
    assert result['recovery_outcome']=='target_not_reached'
    assert not result['recovered']


def test_aggregate():
    eps=[]
    for profile,status in [('pdd_only','recovered'),('learned_v310','target_not_reached')]:
        eps.append(dict(map='SSSSS',speed_mps=8.,profile=profile,summary={'status':'arrive_dest'},
            disturbance={'target_offset_m':.2,'target_reached':status=='recovered',
              'recovered':status=='recovered','recovery_time_s':.5 if status=='recovered' else None}))
    a=aggregate(eps)
    assert len(a)==2
    by={z['profile']:z for z in a}
    assert by['pdd_only']['observed_recovery_rate']==1
    assert by['learned_v310']['observed_recovery_rate'] is None


if __name__=='__main__':
    test_settings();test_success();test_target_not_reached();test_aggregate()
    print('Agent 4 V3.11-A offline tests: PASS (synthetic fixture; not MetaDrive)')
