"""Feature and action contracts for Agent 4 V3.3, simulation research only.

15 frozen PDD features + model action + lane displacement/heading + adaptation flag.
Only the residual *model* supplies correction in student evaluation; no hand-coded
lateral/heading steering correction is evaluated as a student.
"""
from __future__ import annotations
import math
from typing import Mapping
import numpy as np

EXTRA_FEATURES = [
    'base_pdd_steer_md', 'planned_lateral_right_m',
    'heading_error_rad', 'adaptive_lookahead',
]
MAX_STEER = 0.35
MAX_STUDENT_RESIDUAL = 0.25


def finite(value, name='value'):
    try:
        x = float(value)
    except (ValueError, TypeError, OverflowError) as e:
        raise ValueError(f'{name} not a float: {value!r}') from e
    if not math.isfinite(x):
        raise ValueError(f'{name} not finite: {value!r}')
    return x


def clip(value, magnitude):
    return max(-float(magnitude), min(float(magnitude), finite(value)))


def feature_names(pdd_names):
    pdd_names = list(pdd_names)
    if len(set(pdd_names)) != len(pdd_names) or set(pdd_names) & set(EXTRA_FEATURES):
        raise ValueError('Duplicate or conflicting input feature names')
    return pdd_names + EXTRA_FEATURES


def state_features(pdd_features: Mapping, pdd_names, base_steer_md,
                   lateral_right_m, heading_rad, adapted):
    names = feature_names(pdd_names)
    values = {k: finite(pdd_features[k], k) for k in pdd_names}
    values.update(base_pdd_steer_md=finite(base_steer_md,'base_pdd_steer_md'),
                  planned_lateral_right_m=finite(lateral_right_m,'planned_lateral_right_m'),
                  heading_error_rad=finite(heading_rad,'heading_error_rad'),
                  adaptive_lookahead=float(bool(adapted)))
    return {k: values[k] for k in names}


def student_steer_md(model, row, names, max_steer=MAX_STEER,
                     max_residual=MAX_STUDENT_RESIDUAL):
    if not 0 < max_steer <= 1 or not 0 < max_residual <= 1:
        raise ValueError('Invalid steering limits')
    x = np.array([[finite(row[k],k) for k in names]], dtype=float)
    delta = clip(float(model.predict(x)[0]), max_residual)
    base = finite(row['base_pdd_steer_md'], 'base')
    return {'base':base,'learned_residual':delta,
            'steer':clip(base+delta, max_steer)}


def seed_sets(train, val, evaluation):
    a,b,c = [list(z) for z in (train,val,evaluation)]
    if not a or not b or not c or any(int(s)<0 for v in (a,b,c) for s in v):
        raise ValueError('Non-empty nonnegative seed groups required')
    if any(len(set(v))!=len(v) for v in (a,b,c)):
        raise ValueError('Duplicate seed in split')
    if set(a)&set(b) or set(a)&set(c) or set(b)&set(c):
        raise ValueError('Seed group overlap: data leakage')
    return {'train':a,'val':b,'eval':c}
