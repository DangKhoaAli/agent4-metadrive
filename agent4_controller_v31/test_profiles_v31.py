"""Unit tests for profile reproducibility. No MetaDrive required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'agent4_controller_v30'))
from residual_controller_v30 import steering_action
from profiles_v31 import (PROFILES, DEFAULT_PROFILES, select_profiles,
                          options_for, episode_summary)


def test():
    names=select_profiles(DEFAULT_PROFILES)
    assert 'baseline_3' in names and 'baseline_2' in names and 'v30_3' in names
    assert len(names)==len(set(names))
    for invalid in ('', 'unknown', 'v30_3,v30_3'):
        try:
            select_profiles(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Should reject {invalid!r}')
    b=options_for(PROFILES['baseline_3'])
    r=options_for(PROFILES['v30_3'])
    assert b.speed_limit_mps==r.speed_limit_mps==3.0
    assert (b.kp_lateral,b.kp_heading,b.residual_limit)==(.035,.3,.08)
    assert (r.kp_lateral,r.kp_heading,r.residual_limit)==(.035,.3,.08)
    assert b.steer_gain==r.steer_gain==1.0
    assert b.guard_road_margin_m==r.guard_road_margin_m==0.0
    low=options_for(PROFILES['baseline_2'])
    assert low.speed_limit_mps==2.0 and low.steer_gain==b.steer_gain
    assert options_for(PROFILES['heading_3']).kp_lateral==r.kp_lateral
    assert options_for(PROFILES['lateral_3']).kp_heading==r.kp_heading
    assert options_for(PROFILES['balanced_3']).residual_limit<=0.35
    model_carla=.037753857359
    a=steering_action(model_carla,'baseline',kp_lateral=b.kp_lateral,
                      kp_heading=b.kp_heading,residual_limit=b.residual_limit)
    c=steering_action(model_carla,'residual',lateral_right_m=-.40333,
                      heading_error_rad=.13695,kp_lateral=r.kp_lateral,
                      kp_heading=r.kp_heading,residual_limit=r.residual_limit)
    assert abs(a['steer']+.037753857359)<1e-10
    assert c['steer']<a['steer']  # Rightward corrective steer.
    sample=episode_summary({'status':'out_of_road','end_info':{'route_completion':0.35},
        'trace':[{'step':273,'model_steer_md':-0.03}],
        'terminal_road_state':{'active_reasons':['on_yellow_continuous_line']},
        'errors':[]})
    assert sample['terminal_reasons']==['on_yellow_continuous_line']
    assert sample['route_completion']==.35
    print('Agent 4 V3.1 profile and steering tests: PASS')

if __name__=='__main__': test()
