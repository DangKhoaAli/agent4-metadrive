"""Pure contracts for V3.6 learned longitudinal control (MetaDrive simulation)."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_robustness_v34'))
from benchmark_v34 import parse_seeds,parse_speeds,parse_simple_csv
sys.path.insert(0,str(ROOT/'agent4_learned_residual_v33'))
from common_v33 import finite

MAPS=('S','C','X')
EXTRA=('planned_lateral_right_m','heading_error_rad','adaptive_lookahead',
       'target_speed_mps','remaining_route_m','speed_error_mps')
VERSION='V36_IDM_LEARNED_LONGITUDINAL'


def names(pdd_names):
    pdd_names=list(pdd_names)
    if len(set(pdd_names))!=len(pdd_names) or set(pdd_names)&set(EXTRA):
        raise ValueError('Duplicate feature names')
    if 'speed_mps' not in pdd_names:raise ValueError('PDD features need speed_mps')
    return pdd_names+list(EXTRA)


def make_row(features,pdd_names,geometry,adapted,target_speed_mps,remaining_route_m):
    speed=finite(features['speed_mps'],'speed_mps')
    target=finite(target_speed_mps,'target_speed_mps')
    remaining=finite(remaining_route_m,'remaining_route_m')
    if not 0.1<=target<=5.5:raise ValueError(f'Bad target speed: {target}')
    if remaining < -10:raise ValueError(f'Bad remaining distance: {remaining}')
    values={k:finite(features[k],k) for k in pdd_names}
    values.update(planned_lateral_right_m=finite(geometry['lateral_right_m']),
                  heading_error_rad=finite(geometry['heading_error_rad']),
                  adaptive_lookahead=float(bool(adapted)),
                  target_speed_mps=target,remaining_route_m=max(0.0,remaining),
                  speed_error_mps=target-speed)
    return {k:values[k] for k in names(pdd_names)}


def speed_schedule(base_speed_mps,step,enable=True):
    """Diverse *desired speeds* for IDM (native IDM generates the action).

    There is no hand-coded throttle/brake target: we set IDM's desired speed,
    as a navigation command. Every 240 ticks: cruise 90, slow 70, cruise 80.
    """
    speed=finite(base_speed_mps,'base speed')
    if not 0.5<=speed<=5:raise ValueError('Base speed outside 0.5..5')
    if step<0:raise ValueError('Negative step')
    phase=int(step)%240
    if enable and 90<=phase<160:
        return max(0.6,speed*0.55)
    return speed


def check_splits(train,val,heldout):
    groups=[list(z) for z in (train,val,heldout)]
    if any(not a or len(set(a))!=len(a) or min(a)<0 for a in groups):
        raise ValueError('Need distinct nonnegative train/val/eval seeds')
    tr,va,te=map(set,groups)
    if tr&va or tr&te or va&te:raise ValueError('Seed leakage between splits')
    if (tr|va)&set(range(60)):
        raise ValueError('0..59 used for prior steering development/evaluation; do not train/validate V3.6 on these')
    return {'train':groups[0],'val':groups[1],'eval':groups[2]}


def check_eval_seeds(bundle,seeds):
    used=set(bundle['splits']['train'])|set(bundle['splits']['val'])
    if set(seeds)&used:raise ValueError(f'Evaluation seed leakage: {sorted(set(seeds)&used)}')
    if set(seeds)&set(range(60)):
        raise ValueError('Seeds 0..59 were previously evaluated; use fresh V3.6 evaluation seeds')
    return True


def predict_action(bundle,row):
    expected=list(bundle['feature_order'])
    x=np.array([[finite(row[k],k) for k in expected]],dtype=float)
    action=finite(bundle['model'].predict(x)[0],'model longitudinal')
    return float(np.clip(action,-1.0,1.0))


def status_from(info,terminated,truncated):
    if info.get('arrive_dest'):return 'arrive_dest'
    if info.get('out_of_road'):return 'out_of_road'
    if info.get('crash'):return 'crash'
    return 'terminated_or_truncated' if terminated or truncated else 'step_limit'
