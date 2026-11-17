"""Version 2.2: verify right/left axes and preview corrected Agent4 outputs.

Does NOT send learned actions to MetaDrive. Uses a fixed neutral-steering
throttle only to collect observations. Reports the route planning gap for
follow-up lookahead implementation; it does NOT silently alter waypoint horizon.

Usage from the dataset directory:
 python agent4_calibration_v22/probe_shadow_v22.py \
   --model pdd_v2_models/at_t/model_v2.joblib \
   --md-report metadrive_calibration_v21.json \
   --pdd-report pdd_calibration_v21.json \
   --out agent4_v22_shadow.json
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
from coordinate_patch_v22 import (
    determine_local_y_to_right_multiplier,
    correct_features_to_pdd_right,
    carla_to_metadrive_steer,
)

# Directory of original V2 scripts (sibling of this folder).
V2_DIR = Path(__file__).resolve().parent.parent / 'agent4_pdd_v2'
if not (V2_DIR / 'shared_adapter_v2.py').is_file():
    raise RuntimeError(f'Missing original V2 shared adapter: {V2_DIR}')
sys.path.insert(0, str(V2_DIR))
from shared_adapter_v2 import FEATURES, metadrive_features  # noqa: E402


def safe_name(obj) -> str:
    try:
        return str(obj)[:200]
    except Exception:
        return type(obj).__name__


def summarize_route(nav, max_items=5):
    """Inspect *possible* route-aware lane sources, without assuming a 0.4.3 API."""
    out = {}
    for attr in ('checkpoints', '_target_checkpoints_index', 'current_ref_lanes',
                 'next_ref_lanes', 'final_road', 'current_road'):
        try:
            val = getattr(nav, attr)
        except Exception as exc:
            out[attr] = {'read_error': type(exc).__name__ + ': ' + str(exc)[:100]}
            continue
        if val is None:
            out[attr] = None
            continue
        if isinstance(val, (list, tuple)):
            info = {'count': len(val), 'sample': []}
            for item in val[:max_items]:
                lane_data = {'type': type(item).__name__, 'repr': safe_name(item)}
                for k in ('length', 'width', 'index'):
                    try:
                        v = getattr(item, k)
                        if isinstance(v, (int, float)):
                            lane_data[k] = float(v)
                        elif v is not None:
                            lane_data[k] = safe_name(v)
                    except Exception:
                        pass
                info['sample'].append(lane_data)
            out[attr] = info
        elif isinstance(val, (int, float, str, bool)):
            out[attr] = val
        else:
            out[attr] = {'type': type(val).__name__, 'repr': safe_name(val)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--md-report', required=True, type=Path)
    p.add_argument('--pdd-report', required=True, type=Path)
    p.add_argument('--out', type=Path, default=Path('agent4_v22_shadow.json'))
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--steps', type=int, default=12)
    args = p.parse_args()
    md = json.loads(args.md_report.read_text(encoding='utf-8-sig'))
    pdd = json.loads(args.pdd_report.read_text(encoding='utf-8-sig'))
    if md.get('version') != 'v2.1' or pdd.get('status') != 'PDD_WAYPOINT_STEER_DIAGNOSTIC_NOT_COORDINATE_CERTIFICATE':
        raise RuntimeError('Unexpected calibration input files')
    bundle = joblib.load(args.model)
    if list(bundle.get('features', [])) != FEATURES:
        raise RuntimeError('Original trained model features do not match V2 adapter')
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    if not seeds or min(seeds)<0 or args.steps<1:
        p.error('Need nonnegative seeds and --steps >=1')
    from metadrive import MetaDriveEnv
    env = MetaDriveEnv(dict(use_render=False, map='S', traffic_density=0.0,
                            num_scenarios=max(seeds)+1, log_level=40))
    runs, axes = [], []
    try:
        for seed in seeds:
            env.reset(seed=seed)
            for t in range(args.steps):
                sign, frame = determine_local_y_to_right_multiplier(env.agent)
                if t==0:
                    axes.append({'seed':seed, **frame, 'route_inspection':summarize_route(env.agent.navigation)})
                raw, diagnostics = metadrive_features(env)
                corrected = correct_features_to_pdd_right(raw, sign)
                xraw = np.asarray([[raw[f] for f in FEATURES]], dtype=float)
                xnew = np.asarray([[corrected[f] for f in FEATURES]], dtype=float)
                if not (np.isfinite(xraw).all() and np.isfinite(xnew).all()):
                    raise ValueError('Non-finite feature input')
                sraw = float(bundle['models']['steering'].predict(xraw)[0])
                snew = float(bundle['models']['steering'].predict(xnew)[0])
                araw = float(bundle['models']['longitudinal'].predict(xraw)[0])
                anew = float(bundle['models']['longitudinal'].predict(xnew)[0])
                runs.append({
                    'seed':seed, 'step':t, 'lateral_sign':sign,
                    'raw_waypoints':[[raw['wp1_forward_m'],raw['wp1_right_m']],
                                     [raw['wp2_forward_m'],raw['wp2_right_m']]],
                    'corrected_waypoints':[[corrected['wp1_forward_m'],corrected['wp1_right_m']],
                                           [corrected['wp2_forward_m'],corrected['wp2_right_m']]],
                    'raw_side_vehicle_distances':[raw['left_vehicle_dist_m'],raw['right_vehicle_dist_m']],
                    'corrected_side_vehicle_distances':[corrected['left_vehicle_dist_m'],corrected['right_vehicle_dist_m']],
                    'model_steering_before_coordinate_patch_carla':sraw,
                    'model_steering_after_coordinate_patch_carla':snew,
                    'model_steering_sign_only_metadrive_candidate':carla_to_metadrive_steer(snew),
                    'model_longitudinal_before':araw,
                    'model_longitudinal_after':anew,
                    'raw_nav_xy':diagnostics['raw_nav_xy'],
                })
                # Absolutely no learned actions are submitted; straight reference drive.
                _, _, terminated, truncated, info = env.step([0.0, 0.28])
                if terminated or truncated:
                    runs[-1]['reference_episode_end'] = {
                        'arrive_dest': bool(info.get('arrive_dest',False)),
                        'out_of_road': bool(info.get('out_of_road',False)),
                        'crash': bool(info.get('crash',False))}
                    break
    finally:
        env.close()
    left_sign = {r['lateral_sign'] for r in runs}
    yaw = md.get('steering',{}).get('summary',{})
    pdd_sign = pdd.get('steer_vs_yaw',{}).get('sign_agreement',{})
    warn = []
    if len(left_sign)!=1:
        warn.append('Coordinate axis changed between samples: do NOT use adapter yet')
    if pdd_sign.get('n',0) < 100 or pdd_sign.get('same_sign_pct',0) < 95:
        warn.append('PDD steer/yaw sign weak')
    if not yaw.get('bidirectional_sign_supported'):
        warn.append('MetaDrive steer directions not verified')
    if not yaw.get('left_right_direction_consistent_on_2plus_seeds'):
        warn.append('Positive steering physical LEFT/RIGHT requires more complete seeds')
    warn.append('Waypoint lookahead gap not corrected: MetaDrive wp1 is a near checkpoint, PDD wp1 is farther')
    warn.append('Action gain and brake dynamics not matched; sign-only model action is NOT applied')
    report = {
        'status':'V22_COORDINATE_SHADOW_NOT_CLOSED_LOOP',
        'action_model_applied':False, 'reference_action':[0.0,0.28],
        'pdd_sign_stats':pdd.get('steer_vs_yaw'),
        'md_steering_summary':yaw,
        'lateral_signs_found':sorted(left_sign),
        'axis_samples':axes,
        'shadow_samples':runs,
        'warnings':warn,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Axis multipliers (MetaDrive raw lateral -> +right):', sorted(left_sign))
    print('Shadow predictions:',len(runs))
    print('Action applied from model: NO')
    print('Saved:',args.out.resolve())
    for message in warn:
        print('WARNING:', message)

if __name__=='__main__':
    main()
