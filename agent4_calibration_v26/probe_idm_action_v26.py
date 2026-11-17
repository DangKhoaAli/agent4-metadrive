"""Agent 4 V2.6: time-aligned IDM-vs-PDD shadow action audit.

Only MetaDrive IDMPolicy drives. The learned model's predictions are NEVER sent
into env.step. This is a cross-policy diagnostic, NOT a calibrated controller.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder, file in [
    ('agent4_pdd_v2', 'shared_adapter_v2.py'),
    ('agent4_calibration_v22', 'coordinate_patch_v22.py'),
    ('agent4_calibration_v23', 'route_lookahead_v23.py'),
    ('agent4_calibration_v24', 'lane_route_v24.py'),
]:
    parent = ROOT / folder
    if not (parent / file).is_file():
        raise RuntimeError(f'Missing dependency: {parent / file}')
    sys.path.insert(0, str(parent))

from shared_adapter_v2 import FEATURES, metadrive_features  # noqa: E402
from coordinate_patch_v22 import (  # noqa: E402
    determine_local_y_to_right_multiplier,
    correct_features_to_pdd_right,
    carla_to_metadrive_steer,
)
from lane_route_v24 import sample_lane_route_waypoints  # noqa: E402


def stats(xs):
    v = np.asarray([float(x) for x in xs if x is not None and math.isfinite(float(x))], dtype=float)
    if not len(v):
        return None
    return {'n': int(len(v)), 'min': float(v.min()), 'median': float(np.median(v)),
            'p90': float(np.quantile(v, 0.9)), 'max': float(v.max())}


def summarize_action_subset(rows):
    """Only compare actions observed at the same pre-step physical state."""
    matched = [r for r in rows if r.get('usable') and r.get('teacher_action') is not None]
    if not matched:
        return {'n': 0, 'reason': 'No aligned teacher-and-model actions'}
    teacher = np.asarray([r['teacher_action'][0] for r in matched], dtype=float)
    prediction = np.asarray([r['model_sign_only_action'][0] for r in matched], dtype=float)
    teacher_long = np.asarray([r['teacher_action'][1] for r in matched], dtype=float)
    pred_long = np.asarray([r['model_sign_only_action'][1] for r in matched], dtype=float)
    informative = (np.abs(teacher) >= 0.02) & (np.abs(prediction) >= 0.01)
    corr = None
    if len(teacher) >= 4 and np.std(teacher) > 1e-7 and np.std(prediction) > 1e-7:
        corr = float(np.corrcoef(teacher, prediction)[0, 1])
    denom = float(np.dot(prediction, prediction))
    gain = float(np.dot(teacher, prediction) / denom) if denom > 1e-8 else None
    return {
        'n': int(len(matched)),
        'teacher_steer': stats(teacher),
        'model_sign_only_steer': stats(prediction),
        'teacher_longitudinal': stats(teacher_long),
        'model_longitudinal_as_is': stats(pred_long),
        'steering_mae_without_gain': float(np.mean(np.abs(teacher - prediction))),
        'steering_rmse_without_gain': float(np.sqrt(np.mean((teacher - prediction)**2))),
        'longitudinal_mae_raw_not_dynamics_matched': float(np.mean(np.abs(teacher_long - pred_long))),
        'steering_pearson': corr,
        'n_informative_steer_sign': int(np.sum(informative)),
        'informative_steer_sign_agreement': float(np.mean(np.sign(teacher[informative]) == np.sign(prediction[informative]))) if np.any(informative) else None,
        'gain_least_squares_through_zero_DIAGNOSTIC_ONLY': gain,
        'teacher_brake_samples': int(np.sum(teacher_long < -0.02)),
        'model_negative_longitudinal_samples': int(np.sum(pred_long < -0.02)),
    }


def check_recording(before_count, after_count, action):
    if after_count != before_count + 1:
        raise RuntimeError(f'IDM action not recorded once for step: {before_count} -> {after_count}')
    if not isinstance(action, (list, tuple, np.ndarray)) or len(action) != 2:
        raise RuntimeError(f'Unexpected IDM action: {action!r}')
    a = [float(x) for x in action]
    if not all(math.isfinite(x) for x in a):
        raise RuntimeError(f'Non-finite IDM action: {a}')
    return a


def predict(model, feature):
    x = np.asarray([[feature[k] for k in FEATURES]], dtype=float)
    if not bool(np.isfinite(x).all()):
        raise ValueError('Nonfinite input features')
    return [float(model['models']['steering'].predict(x)[0]),
            float(model['models']['longitudinal'].predict(x)[0])]


def get_policy(env):
    # The public example reads the agent's policy through the engine.
    for key in (getattr(env.agent, 'id', None), getattr(env.agent, 'name', None)):
        if key is None:
            continue
        try:
            obj = env.engine.get_policy(key)
            if obj is not None:
                return obj
        except (KeyError, AttributeError, TypeError):
            pass
    raise RuntimeError('Cannot retrieve ego policy through engine.get_policy(agent.id/name)')


def one_map(map_name, seeds, max_steps, horizons, max_gap, model):
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy

    class RecordedIDMPolicy(IDMPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_recorded_action = None
            self.record_count = 0

        def act(self, *args, **kwargs):
            action = super().act(*args, **kwargs)
            self.last_recorded_action = [float(v) for v in action]
            self.record_count += 1
            return action

    # Self-driving reference and explicit instrumented policy, no monkeypatching.
    env = None
    samples, episodes, errors = [], [], []
    try:
        env = MetaDriveEnv(dict(use_render=False, map=map_name,
                                num_scenarios=max(seeds) + 1,
                                traffic_density=0.0,
                                agent_policy=RecordedIDMPolicy,
                                log_level=40,
                                horizon=max(1000, max_steps + 10)))
        for seed in seeds:
            episode_errors, done_info, count = 0, None, 0
            try:
                env.reset(seed=seed)
                for t in range(max_steps):
                    agent = env.agent
                    lane = getattr(agent, 'lane', None)
                    lane_type = type(lane).__name__ if lane is not None else 'None'
                    index = getattr(lane, 'index', None)
                    tokens = [str(index[0]), str(index[1])] if isinstance(index, (list, tuple)) and len(index) >= 2 else []
                    in_curve = 'circular' in lane_type.lower() or 'arc' in lane_type.lower()
                    in_intersection = any('X' in s for s in tokens)
                    row = {'seed': seed, 'step': t, 'speed_mps': float(agent.speed_km_h) / 3.6,
                           'lane_type': lane_type, 'road_nodes': tokens,
                           'in_curve_lane': in_curve, 'in_intersection_block': in_intersection,
                           'usable': False, 'error': None,
                           'teacher_action': None, 'model_sign_only_action': None}
                    policy = get_policy(env)
                    if not isinstance(policy, RecordedIDMPolicy):
                        raise RuntimeError(f'Expected RecordedIDMPolicy, got {type(policy).__name__}')
                    count_before = policy.record_count
                    try:
                        sign, _ = determine_local_y_to_right_multiplier(agent)
                        f0, _ = metadrive_features(env)
                        f = correct_features_to_pdd_right(f0, sign)
                        waypoints, diag = sample_lane_route_waypoints(env, horizons, sign, max_gap)
                        row['route_complete'] = bool(diag.get('complete'))
                        row['lane_chain_continuous'] = bool(diag.get('lane_chain_continuous'))
                        row['route_achieved_m'] = [a['achieved_m'] for a in diag.get('route_sample_statuses', [])]
                        row['lane_indices_along_route'] = diag.get('lane_indices')
                        if bool(diag.get('usable_for_shadow')):
                            f.update(dict(wp1_forward_m=float(waypoints[0][0]), wp1_right_m=float(waypoints[0][1]),
                                          wp2_forward_m=float(waypoints[1][0]), wp2_right_m=float(waypoints[1][1])))
                            s, a = predict(model, f)
                            row['model_carla_action'] = [s, a]
                            row['model_sign_only_action'] = [carla_to_metadrive_steer(s), a]
                            row['waypoints'] = [list(w) for w in waypoints]
                            row['usable'] = True
                    except Exception as exc:
                        row['error'] = f'{type(exc).__name__}: {exc}'
                        if len(errors) < 20:
                            errors.append({'map': map_name, 'seed': seed, 'step': t,
                                           'scope': 'feature_or_geometry', 'error': row['error']})
                        episode_errors += 1
                    # Environment drives using the instrumented IDMPolicy.
                    # Placeholder action has NO effect on the ego vehicle.
                    _, _, term, trunc, info = env.step([0.0, 0.0])
                    policy_after = get_policy(env)
                    try:
                        row['teacher_action'] = check_recording(count_before,
                                                                policy_after.record_count,
                                                                policy_after.last_recorded_action)
                    except Exception as exc:
                        row['error'] = f'IDM_ACTION_CAPTURE_FAILED: {type(exc).__name__}: {exc}'
                        row['usable'] = False
                        if len(errors) < 20:
                            errors.append({'map': map_name, 'seed': seed, 'step': t,
                                           'scope': 'teacher_action_capture', 'error': row['error']})
                        episode_errors += 1
                    samples.append(row)
                    count += 1
                    if term or trunc:
                        done_info = {'terminated': bool(term), 'truncated': bool(trunc),
                                     'arrive_dest': bool(info.get('arrive_dest', False)),
                                     'out_of_road': bool(info.get('out_of_road', False)),
                                     'crash': bool(info.get('crash', False))}
                        break
            except Exception as exc:
                if len(errors) < 20:
                    errors.append({'map': map_name, 'seed': seed, 'step': count,
                                   'scope': 'fatal_episode', 'error': f'{type(exc).__name__}: {exc}'})
            episodes.append({'seed': seed, 'steps': count, 'end_info': done_info,
                             'feature_capture_errors': episode_errors})
    except Exception as exc:
        errors.append({'map': map_name, 'scope': 'init', 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                errors.append({'map': map_name, 'scope': 'close', 'error': f'{type(exc).__name__}: {exc}'})
    good = [r for r in samples if r.get('usable') and r.get('teacher_action') is not None]
    results = {
        'all_usable': summarize_action_subset(good),
        'curve_usable': summarize_action_subset([r for r in good if r['in_curve_lane']]),
        'intersection_usable': summarize_action_subset([r for r in good if r['in_intersection_block']]),
        'straight_usable': summarize_action_subset([r for r in good if not r['in_curve_lane'] and not r['in_intersection_block']]),
        'moving_fast_usable': summarize_action_subset([r for r in good if r['speed_mps'] >= 5]),
    }
    return {'map': map_name, 'episodes': episodes, 'summary': results,
            'total_samples': len(samples), 'n_usable_and_aligned': len(good),
            'n_incomplete_horizon': sum(r.get('route_complete') is False for r in samples),
            'n_idm_action_not_captured': sum(r.get('teacher_action') is None for r in samples),
            'errors': errors, 'example_rows': {
                'start': samples[:4],
                'curve': [r for r in good if r['in_curve_lane']][:8],
                'intersection': [r for r in good if r['in_intersection_block']][:8],
                'unusable': [r for r in samples if not r['usable']][:6],
            }}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('agent4_v26_idm_action_probe.json'))
    ap.add_argument('--maps', default='C,X')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--steps', type=int, default=450)
    ap.add_argument('--wp1', type=float, default=25.0)
    ap.add_argument('--wp2', type=float, default=45.0)
    ap.add_argument('--max-gap', type=float, default=2.0)
    args = ap.parse_args()
    maps = [a.strip() for a in args.maps.split(',') if a.strip()]
    seeds = [int(a.strip()) for a in args.seeds.split(',') if a.strip()]
    if not maps or not seeds or min(seeds) < 0 or args.steps < 10:
        ap.error('Invalid maps, seeds, or step count')
    if not (0 < args.wp1 < args.wp2 <= 50) or not (0 < args.max_gap <= 30):
        ap.error('Invalid lookahead/max-gap')
    model = joblib.load(args.model)
    if list(model.get('features', [])) != FEATURES or model.get('label_mode') != 'at_t':
        raise ValueError('Expected PDD V2 model with label_mode=at_t and exact FEATURES')
    output = {'status': 'V26_IDM_EXPERT_ACTION_VS_PDD_SHADOW_NO_MODEL_CONTROL',
              'model_action_applied': False,
              'reference_driver': 'RecordedIDMPolicy (subclass of MetaDrive IDMPolicy)',
              'placeholder_env_step_action': [0.0, 0.0],
              'lookahead_m': [args.wp1, args.wp2],
              'maps': maps, 'seeds': seeds, 'results': [],
              'warnings': [
                  'Teacher IDM actions are a reference policy, not the PDD expert labels.',
                  'Predicted steering is sign-converted only; its physical gain is not calibrated.',
                  'Comparing raw longitudinal actions is NOT dynamics calibration.',
                  'Least-squares gain is DIAGNOSTIC ONLY. Never apply it without independent tests.',
                  'Near-destination route-horizon failures remain unresolved.',
                  'Model never controls the MetaDrive vehicle.',
              ]}
    for m in maps:
        r = one_map(m, seeds, args.steps, (args.wp1, args.wp2), args.max_gap, model)
        output['results'].append(r)
        print(f"{m}: {r['n_usable_and_aligned']}/{r['total_samples']} aligned; "
              f"IDM capture missing={r['n_idm_action_not_captured']}; "
              f"errors={len(r['errors'])}")
        print('  curve:', r['summary']['curve_usable']['n'],
              'intersection:', r['summary']['intersection_usable']['n'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Model controls ego: NO')
    print('Saved:', args.out.resolve())


if __name__ == '__main__':
    main()
