"""Offline synthetic checks for V3.4: no MetaDrive or model data required."""
from __future__ import annotations
from math import isclose
from types import SimpleNamespace

from benchmark_v34 import (parse_seeds, parse_speeds, parse_simple_csv,
                           attach_student_predictions, summary_rows, group_scores,
                           capture_v33_speed_and_predictions, v33)


def raises(fn, *args):
    try: fn(*args)
    except ValueError: return
    raise AssertionError(f'Expected ValueError from {fn.__name__}')


def main():
    assert parse_seeds('32-35,40,42-43') == [32,33,34,35,40,42,43]
    for v in ('', '1,1','3-2','bad','-1','2-300'):
        raises(parse_seeds,v)
    assert parse_speeds('2,3,4.5') == [2.0,3.0,4.5]
    for v in ('0.1','NaN','6','2,2'):
        raises(parse_speeds,v)
    assert parse_simple_csv('S,C',('S','C','X'))==['S','C']
    raises(parse_simple_csv,'S,S',('S','C','X'))

    dummy={'n_applied_steps':2,'trace':[
        {'step':0,'action_applied':[0.05,0.1]},
        {'step':1,'action_applied':[0.06,0.2]}]}
    attr=[{'original_pdd_steer_md':0.01,'learned_residual':0.04,
           'student_proposed_steer_md':0.05},
          {'original_pdd_steer_md':0.02,'learned_residual':0.04,
           'student_proposed_steer_md':0.06}]
    attach_student_predictions(dummy,attr)
    assert dummy['student_predictions_aligned']
    assert dummy['learned_correction_active_steps']==2
    assert isclose(dummy['trace'][0]['steering_attribution_v34']['learned_residual_md'],0.04)
    assert isclose(dummy['trace'][1]['steering_attribution_v34']['actually_applied_steer_md'],0.06)

    old_opts=v33.options_for; old_pred=v33.student_steer_md
    with capture_v33_speed_and_predictions(4.,[]) as _:
        assert isclose(v33.options_for(v33.PROFILES['lateral_3']).speed_limit_mps,4.)
    assert v33.options_for is old_opts and v33.student_steer_md is old_pred

    fake_base={'status':'arrive_dest','route_completion':0.96,'travelled_approx_m':100,
               'n_applied_steps':400,'max_abs_planned_lateral_right_m':0.1,
               'max_abs_heading_error_rad':0.02,'min_estimated_edge_clearance_m':0.5,
               'near_edge_warning_steps':0,'terminal_reasons':[],'error_count':0}
    fake=[{'map':'C','seed':32,'speed_mps':3.0,'profile':'student_v33','summary':fake_base}]
    rows=summary_rows(fake)
    groups=group_scores(rows)
    assert len(rows)==1 and rows[0]['status']=='arrive_dest'
    assert len(groups)==1 and groups[0]['success']==1 and groups[0]['total']==1
    print('Agent 4 V3.4 offline synthetic tests: PASS')


if __name__=='__main__':main()
