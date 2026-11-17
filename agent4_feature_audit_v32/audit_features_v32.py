"""V3.2 feature/route audit of V3.1 closed-loop policies in MetaDrive.

IMPORTANT: This is an instrumentation-only experiment: same V3.0 pilot,
steering action, speed governor, stop criteria and PDD V2 model. The wrapper
captures exact pre-action model inputs without modifying their values.

Place this folder next to agent4_controller_v30 and agent4_controller_v31.
Only run in MetaDrive simulation; not for physical vehicles.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
for folder, name in [('agent4_controller_v30', 'ab_closed_loop_v30.py'),
                     ('agent4_controller_v31', 'profiles_v31.py')]:
    d = ROOT / folder
    if not (d / name).is_file():
        raise RuntimeError(f'Missing sibling dependency: {d / name}')
    sys.path.insert(0, str(d))

import ab_closed_loop_v30 as v30  # noqa: E402
from profiles_v31 import PROFILES, options_for, select_profiles, episode_summary  # noqa: E402

FEATURES = v30.FEATURES


def capture_episode(env, seed, model, profile_name, *, steps=1400,
                    trace_stride=1, tail_steps=35, pilot=None):
    """Only *observe* feature_for_model and planned_geometry calls.

    A successful pre-action feature/geometry call corresponds to step index t
    because V3.0 calls each exactly once per iteration until the first stop.
    Restore original functions in finally even if the pilot raises.
    """
    profile = PROFILES[profile_name]
    opts = options_for(profile, steps=steps, trace_stride=trace_stride, tail_steps=tail_steps)
    real_feature = v30.feature_for_model
    real_geometry = v30.planned_geometry
    seen_features = []
    seen_geom = []

    def observe_features(sim, max_gap):
        features, route = real_feature(sim, max_gap)
        # Plain copied values only; no modification to the actual model input.
        row = {
            'features_pdd_v2': {k: float(features[k]) for k in FEATURES},
            'route_lane_indices': [repr(x) for x in route.get('lane_indices', [])],
            'remaining_route_m': float(route['remaining_m']),
            'effective_lookahead_m': list(route['effective_lookahead_m']),
            'adaptive_lookahead': bool(route['adapted']),
        }
        seen_features.append(row)
        return features, route

    def observe_geometry(car, max_gap):
        geom = real_geometry(car, max_gap)
        seen_geom.append({
            'planned_lane': str(geom['lane']),
            'planned_lane_index': repr(geom['lane_index']),
            'current_vehicle_lane': str(geom['current_vehicle_lane']),
            'tangent_rad': float(geom['tangent_rad']),
            'local_lane_lateral_to_right_sign': int(geom['local_lane_lateral_to_right_sign']),
        })
        return geom

    v30.feature_for_model = observe_features
    v30.planned_geometry = observe_geometry
    try:
        result = (pilot or v30.pilot_episode)(env, seed, model, profile.mode, opts)
    finally:
        v30.feature_for_model = real_feature
        v30.planned_geometry = real_geometry

    for sample in result.get('trace', []):
        i = int(sample['step'])
        sample['feature_audit'] = seen_features[i] if i < len(seen_features) else None
        sample['geometry_audit'] = seen_geom[i] if i < len(seen_geom) else None
        if sample.get('planned_lateral_right_m') is not None:
            raw = (profile.kp_lateral * float(sample['planned_lateral_right_m'])
                   - profile.kp_heading * float(sample['heading_error_rad']))
            sample['raw_residual_unclipped'] = raw if profile.mode == 'residual' else 0.0
            sample['residual_limit_active'] = bool(profile.mode == 'residual' and
                                                   abs(raw) >= profile.residual_limit - 1e-9)
            base = float(sample['model_steer_md'])
            residual = float(sample['residual_steer'])
            sample['model_and_residual_opposed'] = bool(profile.mode == 'residual' and
                                                        base * residual < 0 and
                                                        abs(base) >= .02 and abs(residual) >= .02)

    trace = result.get('trace', [])
    result['audit'] = {
        'n_feature_calls': len(seen_features),
        'n_geometry_calls': len(seen_geom),
        'n_applied_steps': result.get('n_applied_steps'),
        'n_trace_points_with_features': sum(r['feature_audit'] is not None for r in trace),
        'n_trace_points_with_geometry': sum(r['geometry_audit'] is not None for r in trace),
        'n_residual_limit_points_in_saved_trace': sum(r.get('residual_limit_active', False) for r in trace),
        'n_model_residual_disagreement_points_in_saved_trace': sum(r.get('model_and_residual_opposed', False) for r in trace),
        'last_info_route_completion': trace[-1].get('route_completion') if trace else None,
        'audit_alignment_ok': (len(seen_features) >= int(result.get('n_applied_steps', 0))
                               and len(seen_geom) >= int(result.get('n_applied_steps', 0))
                               and all(r['feature_audit'] is not None and r['geometry_audit'] is not None for r in trace)),
    }
    return result


def reference_comparison(ref_json, map_name, seed, profile_name, episode):
    if ref_json is None:
        return None
    for block in ref_json.get('episodes_by_map', []):
        if block.get('map') != map_name:
            continue
        for r in block.get('episodes', []):
            if r.get('seed') == seed and r.get('profile') == profile_name:
                old = r['summary']
                s = episode_summary(episode)
                delta = abs(float(s['travelled_approx_m']) - float(old['travelled_approx_m']))
                return {'found': True, 'reference_status': old['status'],
                        'status_matches': old['status'] == s['status'],
                        'steps_match': old['n_applied_steps'] == s['n_applied_steps'],
                        'distance_difference_m': delta,
                        'near_exact_reproduction': bool(old['status'] == s['status']
                                                        and old['n_applied_steps'] == s['n_applied_steps']
                                                        and delta < 0.1)}
    return {'found': False}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--maps', default='C,X')
    p.add_argument('--seeds', default='2')
    p.add_argument('--profiles', default='baseline_3,lateral_3')
    p.add_argument('--steps', type=int, default=1400)
    p.add_argument('--trace-stride', type=int, default=1)
    p.add_argument('--tail-steps', type=int, default=35)
    p.add_argument('--reference', type=Path, default=None,
                   help='Optional V3.1 holdout JSON to check deterministic reproducibility')
    p.add_argument('--out', type=Path, default=Path('agent4_v32_feature_audit_CX2.json'))
    a = p.parse_args()
    try:
        a.maps = [s.strip() for s in a.maps.split(',') if s.strip()]
        a.seeds = [int(s.strip()) for s in a.seeds.split(',') if s.strip()]
        a.profiles = select_profiles(a.profiles)
        if not a.maps or any(m not in {'S','C','X'} for m in a.maps):
            raise ValueError('maps must contain S, C or X')
        if not a.seeds or any(s < 0 for s in a.seeds):
            raise ValueError('seeds must be nonnegative integers')
        if any(len(set(xs)) != len(xs) for xs in (a.maps, a.seeds, a.profiles)):
            raise ValueError('duplicate map/seed/profile')
        options_for(PROFILES[a.profiles[0]], steps=a.steps,
                    trace_stride=a.trace_stride, tail_steps=a.tail_steps)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    return a


def main():
    a = parse_args()
    bundle = joblib.load(a.model)
    if list(bundle.get('features', [])) != FEATURES or bundle.get('label_mode') != 'at_t':
        raise ValueError('Wrong bundle: expected PDD V2, label_mode=at_t')
    ref = json.loads(a.reference.read_text(encoding='utf-8')) if a.reference else None
    from metadrive.envs.metadrive_env import MetaDriveEnv

    output = {
        'status': 'V32_TARGETED_PREACTION_FEATURE_AUDIT_NO_POLICY_CHANGE',
        'model_path': str(a.model),
        'maps': a.maps, 'seeds': a.seeds, 'profiles': a.profiles,
        'feature_names': FEATURES, 'steps_limit': a.steps,
        'speed_governor_unchanged': True, 'idm_used': False,
        'learned_steering_applied': True, 'learned_longitudinal_applied': False,
        'policy_changes_from_v31': False, 'traffic_density': 0.0,
        'episodes_by_map': [],
        'warnings': [
            'Simulation only: not for physical vehicles.',
            'All learned inputs are pre-action and are logged for diagnosis; NO new model is trained.',
            'V3.1 guard and action policy are unchanged, including guard_road_margin_m=0.',
            'Geometric edge margin is an approximation; MetaDrive out_of_road flags are authoritative.',
            'Trace alignment must be true; otherwise do not trust logged features.',
            'Only arrive_dest counts as success; guard stop is not a MetaDrive out_of_road failure.',
        ],
    }
    for map_name in a.maps:
        episodes=[]
        env=None
        try:
            env = MetaDriveEnv(dict(use_render=False, map=map_name,
                                    num_scenarios=max(a.seeds)+1,
                                    traffic_density=0.0,
                                    horizon=max(a.steps+10,1000),log_level=40))
            for seed in a.seeds:
                for name in a.profiles:
                    episode = capture_episode(env, seed, bundle, name,
                                              steps=a.steps,
                                              trace_stride=a.trace_stride,
                                              tail_steps=a.tail_steps)
                    summary = episode_summary(episode)
                    replication = reference_comparison(ref,map_name,seed,name,episode)
                    print(f'map={map_name} seed={seed} profile={name} '
                          f'status={summary["status"]} steps={summary["n_applied_steps"]} '
                          f'distance~{summary["travelled_approx_m"]:.1f}m '
                          f'feature_alignment={episode["audit"]["audit_alignment_ok"]} '
                          f'ref_reproduced={replication["near_exact_reproduction"] if replication and replication["found"] else "n/a"}')
                    episodes.append({'seed':seed, 'profile':name,
                                     'settings':asdict(PROFILES[name]),
                                     'summary':summary, 'audit':episode['audit'],
                                     'reference_comparison':replication,
                                     'errors':episode.get('errors', []),
                                     'trace':episode.get('trace', [])})
        finally:
            if env is not None:
                env.close()
        output['episodes_by_map'].append({'map':map_name,'episodes':episodes})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('Saved:',a.out.resolve())
    print('V3.2 captures model inputs; it does NOT change controls, gains, route, guards or MetaDrive policies.')

if __name__=='__main__':
    main()
