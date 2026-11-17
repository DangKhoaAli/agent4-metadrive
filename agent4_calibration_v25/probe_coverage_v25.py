"""Agent 4 V2.5: observe lane- vs road-waypoint predictions while an IDM expert drives.

No learned model action is ever applied to the MetaDrive environment.
The reference vehicle is driven by MetaDrive's built-in IDMPolicy. Designed for
MetaDrive 0.4.3, using shared adapters and geometry samplers from V2.2–V2.4.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import median

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
DEPS = [
    (HERE.parent / 'agent4_pdd_v2', 'shared_adapter_v2.py'),
    (HERE.parent / 'agent4_calibration_v22', 'coordinate_patch_v22.py'),
    (HERE.parent / 'agent4_calibration_v23', 'route_lookahead_v23.py'),
    (HERE.parent / 'agent4_calibration_v24', 'lane_route_v24.py'),
]
for folder, fname in DEPS:
    if not (folder / fname).is_file():
        raise RuntimeError(f'Missing required dependency: {folder / fname}')
    sys.path.insert(0, str(folder))

from shared_adapter_v2 import FEATURES, metadrive_features  # noqa: E402
from coordinate_patch_v22 import (  # noqa: E402
    determine_local_y_to_right_multiplier, correct_features_to_pdd_right,
    carla_to_metadrive_steer,
)
from route_lookahead_v23 import sample_route_waypoints  # noqa: E402
from lane_route_v24 import sample_lane_route_waypoints  # noqa: E402


def descriptive(values):
    if not values:
        return None
    a = sorted(map(float, values))
    return {
        'n': len(a), 'min': a[0], 'median': float(median(a)),
        'p90': float(np.quantile(a, .9)), 'max': a[-1],
    }


def wrapped_delta_rad(new, old):
    return math.atan2(math.sin(new - old), math.cos(new - old))


def road_tokens(index):
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        return [], str(index)
    return [str(index[0]), str(index[1])], str(tuple(index))


def is_curve_lane(lane_type):
    return 'circular' in lane_type.lower() or 'arc' in lane_type.lower()


def in_special_block(tokens, letter):
    return any(re.match(r'^\d+' + re.escape(letter) + r'\d', part) is not None for part in tokens)


def predict(bundle, features):
    arr = np.asarray([[features[k] for k in FEATURES]], dtype=float)
    if not bool(np.isfinite(arr).all()):
        raise ValueError('Non-finite feature values')
    models = bundle['models']
    return float(models['steering'].predict(arr)[0]), float(models['longitudinal'].predict(arr)[0])


def make_feature_variant(base, waypoints):
    if len(waypoints) != 2:
        raise ValueError('Two waypoints required')
    result = dict(base)
    result.update({
        'wp1_forward_m': float(waypoints[0][0]),
        'wp1_right_m': float(waypoints[0][1]),
        'wp2_forward_m': float(waypoints[1][0]),
        'wp2_right_m': float(waypoints[1][1]),
    })
    return result


def select_examples(records, count=8):
    """Retain meaningful cases rather than output thousands of heavy geometry records."""
    chosen = []
    seen = set()
    for r in records:
        if r['geometry_error'] or r['turn_motion'] or r['in_curve_lane'] or r['in_intersection_block']:
            key = (r['seed'], r['step'])
            if key not in seen:
                chosen.append(r)
                seen.add(key)
            if len(chosen) >= count:
                break
    # also capture first and late examples for sanity of journey progress
    if records:
        for r in (records[0], records[-1]):
            key = (r['seed'], r['step'])
            if key not in seen:
                chosen.append(r)
                seen.add(key)
    return chosen[:count + 2]


def run_map(map_name, seeds, steps, lookahead, max_gap, bundle, turn_threshold):
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy

    env = None
    episodes = []
    records = []
    errors = []
    try:
        env = MetaDriveEnv({
            'use_render': False,
            'map': map_name,
            'num_scenarios': max(seeds) + 1,
            'traffic_density': 0.0,
            'agent_policy': IDMPolicy,
            'log_level': 40,
            'horizon': max(1000, steps + 10),
        })
        for seed in seeds:
            count = 0
            end = None
            initial_road = None
            previous_heading = None
            max_speed = 0.0
            traversed = 0.0
            old_pos = None
            failures = 0
            try:
                env.reset(seed=seed)
                for t in range(steps):
                    agent = env.agent
                    pos = tuple(map(float, agent.position[:2]))
                    heading = float(agent.heading_theta)
                    lane = getattr(agent, 'lane', None)
                    lane_type = type(lane).__name__ if lane is not None else 'None'
                    tokens, full_lane_index = road_tokens(getattr(lane, 'index', None))
                    current_road = tuple(tokens)
                    if initial_road is None and tokens:
                        initial_road = current_road
                    delta_heading = wrapped_delta_rad(heading, previous_heading) if previous_heading is not None else 0.0
                    previous_heading = heading
                    if old_pos is not None:
                        traversed += math.hypot(pos[0] - old_pos[0], pos[1] - old_pos[1])
                    old_pos = pos
                    max_speed = max(max_speed, float(agent.speed_km_h) / 3.6)
                    rec = {
                        'seed': seed, 'step': t, 'ego_xy': list(pos),
                        'ego_heading_rad': heading, 'yaw_change_this_step_rad': delta_heading,
                        'speed_mps': float(agent.speed_km_h) / 3.6,
                        'lane_type': lane_type, 'lane_index': full_lane_index,
                        'road_nodes': tokens,
                        'entered_non_initial_road': bool(tokens and current_road != initial_road),
                        'in_curve_lane': is_curve_lane(lane_type),
                        'in_curve_block': in_special_block(tokens, 'C'),
                        'in_intersection_block': in_special_block(tokens, 'X'),
                        'turn_motion': abs(delta_heading) > turn_threshold,
                        'usable': False,
                        'geometry_error': None,
                    }
                    try:
                        sign, _ = determine_local_y_to_right_multiplier(agent)
                        f_raw, _ = metadrive_features(env)
                        f = correct_features_to_pdd_right(f_raw, sign)
                        road_wp, road_diag = sample_route_waypoints(env, lookahead, sign, max_gap)
                        lane_wp, lane_diag = sample_lane_route_waypoints(env, lookahead, sign, max_gap)
                        rec['road_waypoints'] = road_wp
                        rec['lane_waypoints'] = lane_wp
                        rec['road_complete'] = bool(road_diag.get('complete'))
                        rec['road_continuous'] = bool(road_diag.get('geometry_continuous'))
                        rec['lane_complete'] = bool(lane_diag.get('complete'))
                        rec['lane_continuous'] = bool(lane_diag.get('lane_chain_continuous'))
                        rec['lane_indices_along_route'] = lane_diag.get('lane_indices')
                        rec['lane_gaps'] = lane_diag.get('selected_lane_gaps')
                        usable = bool(road_diag.get('usable_for_shadow') and lane_diag.get('usable_for_shadow'))
                        rec['usable'] = usable
                        if usable:
                            s_road, a_road = predict(bundle, make_feature_variant(f, road_wp))
                            s_lane, a_lane = predict(bundle, make_feature_variant(f, lane_wp))
                            rec.update({
                                'steer_road_carla': s_road,
                                'steer_lane_carla': s_lane,
                                'steer_lane_md_sign_only': carla_to_metadrive_steer(s_lane),
                                'longitudinal_road': a_road,
                                'longitudinal_lane': a_lane,
                                'road_lane_lateral_gap_m': abs(road_wp[0][1] - lane_wp[0][1]),
                                'steer_prediction_delta_abs': abs(s_road - s_lane),
                            })
                    except Exception as exc:
                        rec['geometry_error'] = type(exc).__name__ + ': ' + str(exc)
                        failures += 1
                        if len(errors) < 30:
                            errors.append({'seed': seed, 'step': t, 'error': rec['geometry_error']})
                    records.append(rec)
                    count += 1
                    # MetaDrive's built-in IDMPolicy controls the ego; this is a placeholder
                    # input required by env.step(), NOT a learned model action.
                    _obs, _reward, term, trunc, info = env.step([0.0, 0.0])
                    if term or trunc:
                        end = {
                            'terminated': bool(term), 'truncated': bool(trunc),
                            'arrive_dest': bool(info.get('arrive_dest', False)),
                            'out_of_road': bool(info.get('out_of_road', False)),
                            'crash': bool(info.get('crash', False)),
                        }
                        break
            except Exception as exc:
                if len(errors) < 30:
                    errors.append({'seed': seed, 'step': count, 'fatal': type(exc).__name__ + ': ' + str(exc)})
            episodes.append({
                'seed': seed, 'n_samples': count, 'end_info': end,
                'max_speed_mps': max_speed, 'distance_travelled_approx_m': traversed,
                'n_geometry_errors': failures,
            })
    except Exception as exc:
        errors.append({'stage': 'init', 'fatal': type(exc).__name__ + ': ' + str(exc)})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                errors.append({'stage': 'close', 'fatal': type(exc).__name__ + ': ' + str(exc)})

    ok = [r for r in records if r['usable']]
    turning = [r for r in ok if r['turn_motion'] or r['in_curve_lane'] or r['in_curve_block']]
    intersection = [r for r in ok if r['in_intersection_block']]
    summary = {
        'n_samples': len(records),
        'n_usable': len(ok),
        'n_not_usable': len(records) - len(ok),
        'n_in_curve_lane': sum(r['in_curve_lane'] for r in records),
        'n_in_curve_block': sum(r['in_curve_block'] for r in records),
        'n_in_intersection_block': sum(r['in_intersection_block'] for r in records),
        'n_non_initial_road': sum(r['entered_non_initial_road'] for r in records),
        'n_turn_motion_samples': sum(r['turn_motion'] for r in records),
        'n_usable_turning_samples': len(turning),
        'n_usable_intersection_samples': len(intersection),
        'n_lane_chain_broken': sum(r.get('lane_continuous') is False for r in records),
        'n_lane_horizon_incomplete': sum(r.get('lane_complete') is False for r in records),
        'n_road_horizon_incomplete': sum(r.get('road_complete') is False for r in records),
        'road_lane_wp1_lateral_difference_m': descriptive([r['road_lane_lateral_gap_m'] for r in ok]),
        'steer_road_lane_abs_difference': descriptive([r['steer_prediction_delta_abs'] for r in ok]),
        'steer_lane_carla': descriptive([r['steer_lane_carla'] for r in ok]),
        'steer_lane_md_sign_only': descriptive([r['steer_lane_md_sign_only'] for r in ok]),
        'steer_lane_during_turns_carla': descriptive([r['steer_lane_carla'] for r in turning]),
        'steer_lane_during_intersection_carla': descriptive([r['steer_lane_carla'] for r in intersection]),
    }
    warnings = []
    if not records:
        warnings.append('No samples: inspect initialization errors')
    if map_name.upper() == 'C' and not (summary['n_in_curve_lane'] or summary['n_in_curve_block']):
        warnings.append('NO CURVE REACHED: increase --steps and inspect IDM reference motion')
    if map_name.upper() == 'X' and not summary['n_in_intersection_block']:
        warnings.append('NO INTERSECTION REACHED: increase --steps and inspect IDM reference motion')
    if summary['n_lane_chain_broken']:
        warnings.append('Lane-chain discontinuities exist; do NOT consider these predictions valid')
    if summary['n_lane_horizon_incomplete']:
        warnings.append('Insufficient remaining route length for at least one waypoint')
    return {
        'map': map_name, 'reference_policy': 'MetaDrive IDMPolicy (ego)',
        'episodes': episodes, 'summary': summary, 'errors': errors,
        'warnings': warnings,
        'examples': select_examples(records),
        'turn_examples': select_examples([r for r in records if r['turn_motion'] or r['in_curve_lane'] or r['in_curve_block']], 8),
        'intersection_examples': select_examples([r for r in records if r['in_intersection_block']], 8),
        'invalid_examples': [r for r in records if not r['usable']][:6],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('agent4_v25_coverage_probe.json'))
    ap.add_argument('--maps', default='C,X')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--steps', type=int, default=450)
    ap.add_argument('--wp1', type=float, default=25.0)
    ap.add_argument('--wp2', type=float, default=45.0)
    ap.add_argument('--max-gap', type=float, default=2.0)
    ap.add_argument('--turn-deg-per-step', type=float, default=0.5)
    args = ap.parse_args()
    maps = [s.strip() for s in args.maps.split(',') if s.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    if not maps or not seeds or min(seeds) < 0 or args.steps < 10 or not (0 < args.wp1 < args.wp2 <= 50):
        ap.error('Bad maps/seeds/steps/lookahead')
    if not (0 < args.max_gap < 30) or not (0 < args.turn_deg_per_step < 180):
        ap.error('Bad max gap / turn threshold')
    bundle = joblib.load(args.model)
    if list(bundle.get('features', [])) != FEATURES or bundle.get('label_mode') != 'at_t':
        raise ValueError('Expected PDD V2 at_t with exact shared FEATURES order')
    results = []
    for map_name in maps:
        result = run_map(map_name, seeds, args.steps, (args.wp1, args.wp2), args.max_gap,
                         bundle, math.radians(args.turn_deg_per_step))
        results.append(result)
        s = result['summary']
        print(f"{map_name}: {s['n_usable']}/{s['n_samples']} usable; "
              f"curved-lane={s['n_in_curve_lane']}, curve-block={s['n_in_curve_block']}, "
              f"intersection-block={s['n_in_intersection_block']}, "
              f"turn-motion={s['n_turn_motion_samples']}, errors={len(result['errors'])}")
        for w in result['warnings']:
            print('  WARNING:', w)
    report = {
        'status': 'V25_IDM_REFERENCE_COVERAGE_SHADOW_NOT_CLOSED_LOOP',
        'model_action_applied': False,
        'reference_ego_policy': 'MetaDrive IDMPolicy',
        'env_step_placeholder_action_not_used_for_control': [0.0, 0.0],
        'maps': maps, 'seeds': seeds, 'steps_per_seed_requested': args.steps,
        'lookahead_m': [args.wp1, args.wp2],
        'results': results,
        'warnings': [
            'IDMPolicy drives ego, not the learned model; all model predictions are shadow-only.',
            'Physical action gains CARLA-to-MetaDrive remain UNCALIBRATED.',
            'Nearest-continuous lane is not guaranteed to be correct for planned turns or lane changes.',
            'A curved block map label does NOT prove the ego reached that curve; inspect coverage counts.',
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', args.out.resolve())


if __name__ == '__main__':
    main()
