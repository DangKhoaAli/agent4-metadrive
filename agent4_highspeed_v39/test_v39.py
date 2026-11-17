"""V3.9 offline unit tests: contracts, unit conversions, analysis and hook restoration.
Does not run MetaDrive or load the actual user-trained models.
"""
from __future__ import annotations
import math
from benchmark_v39 import (parse_speeds_v39,metrics_from_actions,
                           group_statistics,capture_governor,SIM_STEP_S)
import benchmark_v34 as v34


def run():
    assert parse_speeds_v39('4,6,8,10')==[4.,6.,8.,10.]
    assert parse_speeds_v39('2.5,10.5')==[2.5,10.5]
    for bad in ('', '4,4', 'nan', 'inf', '-1', '13', 'four'):
        try:parse_speeds_v39(bad)
        except ValueError:pass
        else:raise AssertionError(f'Invalid speed accepted: {bad!r}')
    original=v34.v30.governed_speed_action
    records=[]
    with capture_governor(records):
        act,target=v34.v30.governed_speed_action(0.,10.,200.)
        assert target==10. and act==.4
        act,target=v34.v30.governed_speed_action(10.,10.,200.)
        assert abs(act)<1e-10
    assert v34.v30.governed_speed_action is original
    assert len(records)==2 and records[1]['speed_mps_before']==10.
    stat=metrics_from_actions(records,2)
    assert abs(stat['sim_elapsed_s']-2*SIM_STEP_S)<1e-10
    assert stat['max_speed_mps']==10. and stat['throttle_steps']==1
    assert stat['governor_deviation_mae']==0
    try:metrics_from_actions(records,3)
    except RuntimeError:pass
    else:raise AssertionError('Action misalignment not detected')
    invalid=[{**records[0],'front_present':True}]
    try:metrics_from_actions(invalid,1)
    except RuntimeError:pass
    else:raise AssertionError('Traffic present not detected')
    failed=dict(map='S',speed_mps=10.,profile='learned_v38',summary={
        'status':'out_of_road','time_to_goal_s':None,'max_speed_mps':10.})
    passed=dict(map='S',speed_mps=10.,profile='learned_v38',summary={
        'status':'arrive_dest','time_to_goal_s':14.,'max_speed_mps':9.})
    stats=group_statistics([failed,passed]);assert len(stats)==1
    assert stats[0]['success']==1 and stats[0]['total']==2
    assert stats[0]['mean_time_to_goal_success_only_s']==14.
    assert stats[0]['max_observed_speed_mps']==10.
    print('Agent 4 V3.9 offline tests: PASS')

if __name__=='__main__':run()
