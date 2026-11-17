"""Pure V3.8 contracts. MetaDrive-dependent code lives in traffic_v38.py."""
from __future__ import annotations
import math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_onpolicy_longitudinal_v37'))
from common_v37 import names as old_names, validate_new_splits

VERSION='V38_CONTROLLED_LEAD_TRAFFIC_BC'
STATUS='V38_CONTROLLED_LEAD_TEACHER_DATA'
SCENARIOS=('slow_release','stop_release','brake_release')
EXTRA=('front_present','front_gap_m','front_speed_mps','front_closing_mps','front_ttc_s')

def feature_names(pdd_features):
    base=old_names(pdd_features)
    if len(set(base+list(EXTRA)))!=len(base)+len(EXTRA):raise ValueError('Duplicate names')
    return base+list(EXTRA)

def front_features(front):
    # Always finite; absence is explicit so '80 m' cannot be mistaken for observed gap.
    if not front['present']:
        return dict(front_present=0.,front_gap_m=80.,front_speed_mps=0.,
                    front_closing_mps=0.,front_ttc_s=30.)
    gap=float(front['gap_m']); speed=float(front['speed_mps']); closing=float(front['closing_mps'])
    if min(gap,speed,closing)<0 or not all(math.isfinite(x) for x in (gap,speed,closing)):
        raise ValueError('Invalid observed traffic numbers')
    ttc=min(30.,gap/max(closing,1e-3)) if closing>0 else 30.
    return dict(front_present=1.,front_gap_m=min(gap,80.),front_speed_mps=min(speed,30.),
                front_closing_mps=min(closing,30.),front_ttc_s=ttc)

def expert_follow(speed,route_target,front):
    """Hand-designed demonstration teacher, NOT learned, NOT IDM.

    Computes desired speed from observed clearance and closure; bounded longitudinal.
    Never used as a fallback in the V3.8 learned evaluation mode.
    """
    speed=float(speed); route_target=float(route_target)
    if speed<0 or route_target<0:raise ValueError('Bad speed')
    desired=route_target
    if front['present']:
        gap=max(0.,float(front['gap_m']))
        closing=max(0.,float(front['closing_mps']))
        # Smooth space-to-speed governor, progressively lowers speed near a lead.
        safe_speed=max(0., (gap-5.)/2.2-0.30*closing)
        desired=min(desired,safe_speed)
    # More brake authority than the empty-road governor, conservative positive throttle.
    action=float(np.clip(.23*(desired-speed),-.60,.40))
    return action,desired

def validate_splits(groups):
    validate_new_splits(groups)
    for group in groups.values():
        if min(group)<100:raise ValueError('V3.8 training/validation/eval seeds must be >=100')
    return groups

def predict(bundle,row):
    arr=np.array([[float(row[n]) for n in bundle['feature_order']]],dtype=float)
    if not np.isfinite(arr).all():raise ValueError('Non-finite traffic features')
    val=float(bundle['model'].predict(arr)[0])
    if not math.isfinite(val):raise ValueError('Non-finite traffic prediction')
    return float(np.clip(val,-1.,1.))
