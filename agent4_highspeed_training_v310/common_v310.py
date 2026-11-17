"""V3.10 contracts: learned high-speed steering residual, never teacher at inference.

Uses PDD V2's original 15 features + V3.3 4 extra inputs. A smooth
zero-centered model has a structural zero correction when lateral and heading
errors are both zero. This is model architecture, NOT a safety/governor fallback.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_learned_residual_v33'))
from common_v33 import feature_names,state_features,finite,clip

VERSION='V310_LEARNED_HIGH_SPEED_LATERAL_RESIDUAL'
DATA_STATUS='V310_HIGH_SPEED_RULE_LATERAL_SUPERVISION'
MAX_STEER=0.35
MAX_RESIDUAL=0.08
ALLOWED_MAPS=('S','SS','SSS','SSSS','SSSSS')
DEFAULT_SPEEDS=(4.,6.,8.,10.)
BASIS_NAMES=['lateral_right_m','heading_rad','lateral_x_speed','heading_x_speed',
             'lateral_cubed','heading_cubed']


def basis(lateral,heading,speed):
    e=finite(lateral,'lateral_right_m')
    h=finite(heading,'heading_rad')
    v=finite(speed,'speed_mps')
    if not 0<=v<=40:raise ValueError('Vehicle speed outside validated range')
    # Linear and small nonlinear state feedback; NO intercept or speed-only term.
    q=v/10.
    return np.asarray([e,h,e*q,h*q,e**3,h**3],dtype=float)


def student_action(bundle,model_row):
    if bundle.get('version')!=VERSION:
        raise ValueError('Not a V3.10 steering bundle')
    if list(bundle.get('feature_order',[]))!=list(model_row):
        # A dict read from state_features follows feature_order, but assert set
        if set(bundle.get('feature_order',[]))!=set(model_row):
            raise ValueError('Student feature contract mismatch')
    x=basis(model_row['planned_lateral_right_m'],model_row['heading_error_rad'],
            model_row['speed_mps'])
    raw=float(bundle['student_model'].predict(x.reshape(1,-1))[0])
    residual=clip(raw,float(bundle['max_learned_residual']))
    base=finite(model_row['base_pdd_steer_md'],'base_pdd_steer_md')
    action=clip(base+residual,float(bundle['max_steer']))
    return {'base':base,'raw_learned_residual':raw,'learned_residual':residual,
            'steer':action}


def validate_seed_groups(train,val,holdout,lowest=130):
    groups=[list(map(int,x)) for x in (train,val,holdout)]
    if any(not x or len(set(x))!=len(x) for x in groups):
        raise ValueError('Empty or duplicate seed group')
    if min(x for g in groups for x in g)<lowest:
        raise ValueError(f'V3.10 seeds must start at {lowest}; older seeds used in earlier diagnostics')
    if any(set(groups[i])&set(groups[j]) for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Train/validation/evaluation seed overlap')
    return {'train':groups[0],'val':groups[1],'eval':groups[2]}


def parse_seed_spec(spec):
    out=[]
    for item in spec.split(','):
        token=item.strip()
        if not token:continue
        if '-' in token:
            start,end=map(int,token.split('-',1))
            if end<start or end-start>1000:raise ValueError('Invalid seed range')
            out.extend(range(start,end+1))
        else:out.append(int(token))
    if not out or len(out)!=len(set(out)):raise ValueError('Missing or duplicated seeds')
    return out


def parse_values(spec,valid,convert=str):
    vals=[convert(v.strip()) for v in spec.split(',') if v.strip()]
    if not vals or len(vals)!=len(set(vals)) or any(v not in valid for v in vals):
        raise ValueError(f'Invalid values {vals}; allowed {valid}')
    return vals


def perturbation(step,seed,amp,pulse_steps=6):
    """Deterministic ± steering perturbations; teacher label stays unperturbed.
    Four small perturbation windows per episode, two of each sign. Not
    a simulator-based RNG stream. Applies on *steering action*, not features.
    """
    if not 0<=amp<=0.015 or not 1<=pulse_steps<=12:
        raise ValueError('Perturb amplitude 0..0.015 and pulse steps 1..12')
    phase=(step+17*seed)%240
    if 45<=phase<45+pulse_steps:return amp
    if 105<=phase<105+pulse_steps:return -amp
    if 155<=phase<155+pulse_steps:return amp
    if 215<=phase<215+pulse_steps:return -amp
    return 0.


def check_pdd(pdd):
    ROOT=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(ROOT/'agent4_controller_v30'))
    import ab_closed_loop_v30 as v30
    if pdd.get('label_mode')!='at_t' or list(pdd.get('features',[]))!=list(v30.FEATURES):
        raise ValueError('Expected frozen PDD V2 at_t with exactly 15 ordered features')
    return list(pdd['features'])


def check_bundle(bundle,pdd_names,holdout=None):
    if bundle.get('version')!=VERSION or bundle.get('feature_order')!=feature_names(pdd_names):
        raise ValueError('Wrong V3.10 steering model or PDD feature list')
    if bundle.get('basis_names')!=BASIS_NAMES or bundle.get('max_steer')!=MAX_STEER or bundle.get('max_learned_residual')!=MAX_RESIDUAL:
        raise ValueError('V3.10 model action limits / basis mismatch')
    tr=set(bundle.get('train_seeds',[]));va=set(bundle.get('val_seeds',[]))
    if tr&va or (holdout and (set(holdout)&(tr|va))):
        raise ValueError('Evaluation seed leakage')
    return True
