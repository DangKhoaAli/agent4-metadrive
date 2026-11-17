"""V3.15 shared contracts; never mutates or imports training scripts as side effects."""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

VERSION = 'AGENT4_V315_RECOVERY_DAGGER'
STEP_SECONDS = 0.1


def legacy_modules(legacy_root):
    root = Path(legacy_root).resolve()
    pairs = [root / 'agent4_steering_v313', root / 'agent4_idm_dataset_v312b']
    missing = [str(p) for p, f in zip(pairs, ('features_v313.py', 'collect_v312b.py')) if not (p / f).is_file()]
    if missing:
        raise FileNotFoundError('Required original V3.13/V3.12B code missing, WITHOUT modifying it: ' + ', '.join(missing))
    for p in pairs:
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    return importlib.import_module('features_v313'), importlib.import_module('collect_v312b')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def original_manifest(dataset):
    f = Path(dataset) / 'manifest.json'
    doc = json.loads(f.read_text(encoding='utf-8'))
    if doc.get('version') != 'AGENT4_V312B_FULL_STEP_IDM_DATASET':
        raise ValueError('Require the original validated V3.12B dataset manifest')
    groups = doc['config']['split_seeds']
    seeds = {k: set(map(int, groups[k])) for k in ('train', 'val', 'test')}
    if not all(seeds.values()) or (seeds['train'] & seeds['val']) or (seeds['train'] & seeds['test']) or (seeds['val'] & seeds['test']):
        raise ValueError('Seed splits overlap or are empty')
    return doc, seeds, digest(f)


def allowed_train_seeds(chosen, splits):
    chosen = set(map(int, chosen))
    if not chosen or not chosen <= splits['train']:
        raise ValueError(f'Collect ONLY V3.12B TRAIN seeds. Requested={sorted(chosen)}, permitted={sorted(splits["train"])}')
    return sorted(chosen)


def allowed_evaluation_seeds(chosen, splits, mode, frozen=False):
    selected = set(map(int, chosen))
    if not selected: raise ValueError('Empty evaluation seeds')
    if mode == 'validation':
        if not selected <= splits['val']:
            raise ValueError('Validation evaluation requires validation seeds only')
    elif mode == 'final':
        if not frozen: raise ValueError('Final holdout requires --confirm-frozen')
        old = splits['train'] | splits['val'] | splits['test']
        if selected & old: raise ValueError('Final holdout must use NEW seeds, never the previously analyzed test seeds')
    else: raise ValueError('Unknown evaluation mode')
    return sorted(selected)


def check_baseline(bundle, manifest_hash, features_module):
    if (bundle.get('version') != features_module.VERSION or
        tuple(bundle.get('features', ())) != features_module.FEATURES or
        bundle.get('teacher_fallback') is not False):
        raise ValueError('Invalid V3.13 frozen no-teacher-fallback baseline')
    if bundle.get('trained_on_manifest_sha256') != manifest_hash:
        raise ValueError('Baseline model and original dataset manifest do not match')


def bounded(x):
    x = float(x)
    if not math.isfinite(x): raise ValueError('Nonfinite action')
    return min(1.0, max(-1.0, x))


def governor(speed, target, gain=0.23, limit=0.4):
    if not all(math.isfinite(x) for x in (speed, target)) or target <= 0:
        raise ValueError('Invalid longitudinal target')
    return max(-limit, min(limit, gain * (target - speed)))


def recovery_gate(pre):
    """Continuous confidence gate: zero near demonstration manifold, one at strong deviations.

    No privileged teacher data is read at inference; only V3.13 input features.
    """
    lateral = abs(float(pre['planned_lateral_right_m']))
    heading = abs(float(pre['heading_error_rad']))
    magnitude = max(lateral / 0.35, heading / 0.05)
    return max(0.0, min(1.0, (magnitude - 0.30) / 0.70))


def predict(bundle, pre, features_module, alpha=None):
    if bundle.get('version') != VERSION or bundle.get('teacher_fallback') is not False:
        raise ValueError('Not a valid no-fallback V3.15 bundle')
    baseline = features_module.steering_prediction(bundle['baseline'], pre)
    weight = float(bundle.get('alpha', 0.0) if alpha is None else alpha)
    if not math.isfinite(weight) or not 0 <= weight <= 1: raise ValueError('Alpha must be in [0,1]')
    gate = recovery_gate(pre)
    if weight == 0 or gate == 0:
        return baseline, baseline, 0., gate
    x = features_module.row_features(pre).reshape(1, -1)
    residual = float(bundle['residual_model'].predict(x)[0])
    if not math.isfinite(residual): raise ValueError('Nonfinite residual prediction')
    correction = max(-0.4, min(0.4, residual))
    return bounded(baseline + weight * gate * correction), baseline, correction, gate


def read_step(result):
    if len(result) == 5:
        _, _, terminated, truncated, info = result
    elif len(result) == 4:
        _, _, terminated, info = result
        truncated = False
    else: raise RuntimeError('Unknown MetaDrive step signature')
    if not isinstance(info, dict): raise TypeError('Missing MetaDrive info dictionary')
    return bool(terminated), bool(truncated), info


def episode_status(terminated, truncated, info):
    if info.get('arrive_dest'): return 'arrive_dest'
    if info.get('out_of_road'): return 'out_of_road'
    if any(info.get(k) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_sidewalk')): return 'crash'
    if truncated: return 'truncated'
    if terminated: return 'other_termination'
    return 'step_limit'
