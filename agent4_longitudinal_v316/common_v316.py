"""V3.16 shared contracts for IDM-labelled, learned MetaDrive gas/brake.

The V3.13 steering estimator stays frozen. All observation features are pre-step.
Never infer teacher actions at evaluation. No speed governor fallback for learner.
"""
from __future__ import annotations
import hashlib, importlib, math, sys
from pathlib import Path
import numpy as np

VERSION = 'AGENT4_V316_SHADOW_IDM_LONGITUDINAL'
EXTRA = ('target_speed_mps', 'speed_error_mps')
# Seeds 194..201 have ALREADY been inspected for V3.15's final holdout.
USED_PRIOR_FINAL = set(range(194, 202))


def dependencies(legacy_root):
    root=Path(legacy_root).resolve()
    for directory, filename in [('agent4_recovery_v315','common_v315.py'),
                                ('agent4_steering_v313','features_v313.py'),
                                ('agent4_idm_dataset_v312b','collect_v312b.py')]:
        p=root/directory
        if not (p/filename).is_file():
            raise FileNotFoundError(f'V3.16 needs existing {p/filename}; legacy files will not be modified')
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    c=importlib.import_module('common_v315')
    feat, geometry=c.legacy_modules(root)
    return c, feat, geometry


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def speed_schedule(base, step):
    """Explicit desired-speed command: 65 ticks cruise, 55 slow, 60 cruise.
    The low-level gas/brake is NOT computed from this schedule by the student.
    """
    base=float(base)
    if not (math.isfinite(base) and 1 <=base<=12 and int(step)>=0):
        raise ValueError('Invalid base speed/step')
    return max(.65,.55*base) if 65<=int(step)%180<120 else base


def model_features(feat):
    return tuple(feat.FEATURES)+EXTRA


def make_features(row, feat):
    x=list(feat.row_features(row))
    target=float(row['target_speed_mps'])
    speed=float(row['speed_mps'])
    if not all(map(math.isfinite,(target,speed))) or target<.1 or speed<0:
        raise ValueError('Nonfinite or invalid speed features')
    return np.asarray(x+[target,target-speed],dtype=np.float32)


def predict_long(bundle, row, feat):
    if bundle.get('version')!=VERSION or bundle.get('teacher_fallback') is not False:
        raise ValueError('Wrong V3.16 bundle or teacher fallback enabled')
    if tuple(bundle.get('feature_names',()))!=model_features(feat):
        raise ValueError('V3.16 feature schema mismatch')
    y=float(bundle['estimator'].predict(make_features(row,feat).reshape(1,-1))[0])
    if not math.isfinite(y):raise ValueError('Model gas/brake is NaN/Inf')
    return float(np.clip(y,-1.,1.))


def allowable_seeds(chosen, original_splits, split):
    values=set(map(int,chosen))
    if not values:raise ValueError('Empty seeds')
    if split in ('train','val'):
        if not values<=original_splits[split]:raise ValueError(f'{split} seeds must be a subset of original {split} split')
    elif split=='final':
        if values & (original_splits['train']|original_splits['val']|original_splits['test']|USED_PRIOR_FINAL):
            raise ValueError('Final V3.16 holdout MUST be fresh; do not reuse 170..201')
        if min(values)<202:raise ValueError('Use fresh final seeds >=202')
    else:raise ValueError('Unknown split')
    return sorted(values)


def validate_model(bundle, manifest_hash, baseline_hash, feat):
    if bundle.get('version')!=VERSION or bundle.get('teacher_fallback') is not False:
        raise ValueError('Invalid V3.16 model')
    if bundle.get('original_manifest_sha256')!=manifest_hash or bundle.get('baseline_sha256')!=baseline_hash:
        raise ValueError('V3.16 bundle paired with different V3.12B/V3.13 input')
    if tuple(bundle.get('feature_names',()))!=model_features(feat):
        raise ValueError('V3.16 input feature mismatch')


def pair_action(raw):
    try:a=tuple(float(x) for x in raw)
    except (TypeError,ValueError) as exc:raise ValueError('Invalid 2-vector action') from exc
    if len(a)!=2 or not all(map(math.isfinite,a)):
        raise ValueError('Nonfinite or non-pair action')
    return a


def verified_action(vehicle, steer, longitude, tolerance=5e-4):
    got=pair_action(vehicle.current_action)
    expected=(float(steer),float(longitude))
    if max(abs(got[0]-expected[0]),abs(got[1]-expected[1]))>tolerance:
        raise RuntimeError(f'Sent command differs from current_action: {expected} vs {got}')
    if hasattr(vehicle,'steering') and hasattr(vehicle,'throttle_brake'):
        if max(abs(got[0]-float(vehicle.steering)),abs(got[1]-float(vehicle.throttle_brake)))>tolerance:
            raise RuntimeError('current_action differs from actual vehicle controls')
    return got
