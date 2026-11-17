"""Frozen V3.13 steering input contract.

All inputs are pre-action MetaDrive privileged geometry from V3.12B.
No post-action fields, teacher action, seed/map, raw position, or split.
"""
from __future__ import annotations
import math
import numpy as np

VERSION = 'AGENT4_V313_IDM_IMITATION_STEERING'
DATA_VERSION = 'AGENT4_V312B_FULL_STEP_IDM_DATASET'
FEATURES = (
    'speed_mps',
    'planned_lateral_right_m',
    'heading_error_rad',
    'distance_to_lane_end_m',
    'curvature_proxy_rad_per_m',
    'lookahead_heading_5m_rad',
    'lookahead_heading_10m_rad',
    'lookahead_heading_20m_rad',
    'lookahead_5m_available',
    'lookahead_10m_available',
    'lookahead_20m_available',
    'is_circular_lane',
)
LABEL = 'applied_steering'
ZERO_IF_UNAVAILABLE = ('lookahead_heading_5m_rad', 'lookahead_heading_10m_rad', 'lookahead_heading_20m_rad')


def row_features(row):
    """Return numeric vector, refusing unsafe missing essential observations.

    V3.12B clean rows have 5m and 10m lookahead available. At runtime,
    unavailable 20m is represented as 0 with a separate availability flag.
    """
    out=[]
    originally_missing=set()
    for name in FEATURES:
        value=row.get(name)
        if value is None or str(value).strip() == '':
            if name in ZERO_IF_UNAVAILABLE:
                value=0.
                originally_missing.add(name)
            else:
                raise ValueError(f'Missing essential V3.13 input: {name}')
        try: x=float(value)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError(f'Non-numeric feature {name}: {value!r}') from exc
        if not math.isfinite(x):
            raise ValueError(f'Nonfinite feature {name}')
        out.append(x)
    for dist in ('5','10','20'):
        flag=out[FEATURES.index(f'lookahead_{dist}m_available')]
        h=out[FEATURES.index(f'lookahead_heading_{dist}m_rad')]
        if flag not in (0.,1.):raise ValueError(f'Invalid lookahead_{dist}m_available: {flag}')
        if flag == 0. and abs(h)>1e-8:
            raise ValueError(f'Lookahead {dist}m unavailable but has nonzero heading')
        if flag == 1. and f'lookahead_heading_{dist}m_rad' in originally_missing:
            raise ValueError(f'Lookahead {dist}m marked available but its value is missing')
    if out[FEATURES.index('is_circular_lane')] not in (0.,1.):
        raise ValueError('Invalid is_circular_lane')
    return np.asarray(out,dtype=np.float32)


def steering_prediction(bundle,pre):
    if bundle.get('version') != VERSION or tuple(bundle.get('features',()))!=FEATURES:
        raise ValueError('Incompatible V3.13 model/feature schema')
    vec=row_features(pre).reshape(1,-1)
    value=float(bundle['estimator'].predict(vec)[0])
    if not math.isfinite(value):raise ValueError('Steering model emitted NaN/Inf')
    return max(-1.,min(1.,value))
