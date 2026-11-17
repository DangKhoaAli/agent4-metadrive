"""MetaDrive 0.4.3 action/nav calibration; DOES NOT run a learned model.

Each intervention starts from the same seed and warm-up trajectory on a straight
road with traffic_density=0. Yaw response is compared against a zero-steer
control to remove road drift; speed/brake comparisons also include zero-action.

Usage:
 python agent4_calibration_v21/calibrate_metadrive_v21.py --out metadrive_calibration_v21.json
 python agent4_calibration_v21/calibrate_metadrive_v21.py --seeds 0,1,2 --out metadrive_calibration_v21.json
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def numeric(value):
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f'Non-finite value: {val}')
    return val


def angle_delta(a: float, b: float) -> float:
    """Minimal wrapped angle of a-b, in radians."""
    return (numeric(a) - numeric(b) + math.pi) % (2 * math.pi) - math.pi


def median(values):
    arr = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(arr) if arr else None


def extract_state(env):
    agent = env.agent
    pos = agent.position
    return {
        'speed_mps': numeric(agent.speed_km_h) / 3.6,
        'heading_rad': numeric(agent.heading_theta),
        'position_xy': [numeric(pos[0]), numeric(pos[1])],
    }


def step_command(env, steer: float, longitudinal: float):
    _, reward, terminated, truncated, info = env.step([float(steer), float(longitudinal)])
    reason = {
        'terminated': bool(terminated), 'truncated': bool(truncated),
        'arrive_dest': bool(info.get('arrive_dest', False)),
        'out_of_road': bool(info.get('out_of_road', False)),
        'crash': bool(info.get('crash', False)),
    }
    return bool(terminated or truncated), reason


def run_trial(env, seed: int, warmup_steps: int, warmup_long: float,
              command: tuple[float, float], pulse_steps: int):
    env.reset(seed=int(seed))
    terminal = False
    termination = None
    for i in range(warmup_steps):
        terminal, termination = step_command(env, 0.0, warmup_long)
        if terminal:
            break
    at_start = extract_state(env)
    at_end = at_start
    steps_done = 0
    if not terminal:
        for i in range(pulse_steps):
            terminal, termination = step_command(env, command[0], command[1])
            steps_done += 1
            at_end = extract_state(env)
            if terminal:
                break
    # Report yaw relative to initial car heading, not world 0.
    yaw = angle_delta(at_end['heading_rad'], at_start['heading_rad'])
    return {
        'seed': seed, 'warmup_steps_requested': warmup_steps,
        'test_action': [float(command[0]), float(command[1])],
        'pulse_steps_requested': pulse_steps, 'pulse_steps_completed': steps_done,
        'start': at_start, 'end': at_end,
        'yaw_delta_rad': yaw,
        'speed_delta_mps': at_end['speed_mps'] - at_start['speed_mps'],
        'position_delta_xy': [at_end['position_xy'][j] - at_start['position_xy'][j] for j in range(2)],
        'left_displacement_m': (-math.sin(at_start['heading_rad']) * (at_end['position_xy'][0] - at_start['position_xy'][0]) +
                                math.cos(at_start['heading_rad']) * (at_end['position_xy'][1] - at_start['position_xy'][1])),
        'early_end': bool(terminal), 'end_info': termination,
    }


def _lane_point(agent, longitudinal, lateral=0.0):
    lane = agent.lane
    point = lane.position(longitudinal, lateral)
    delta = [float(point[0]) - float(agent.position[0]), float(point[1]) - float(agent.position[1])]
    local_x, local_y = agent.convert_to_local_coordinates(delta, 0.0)
    return [float(local_x), float(local_y)]


def nav_snapshot(env, step: int, seed: int):
    agent = env.agent
    nav = agent.navigation
    info = [float(x) for x in nav.get_navi_info()]
    d = numeric(nav.NAVI_POINT_DIST)
    native = [
        [(info[0] * 2 - 1) * d, (info[1] * 2 - 1) * d],
        [(info[5] * 2 - 1) * d, (info[6] * 2 - 1) * d],
    ]
    lane = agent.lane
    result = {
        'step': step, 'seed': seed, 'speed_mps': numeric(agent.speed_km_h) / 3.6,
        'navi_point_dist_m': d, 'native_wp1_local_xy': native[0],
        'native_wp2_local_xy': native[1],
        'lane_center_at_forward_dist': {},
        'lane_current_local_longitudinal_lateral': None,
        'lane_length_m': None,
    }
    try:
        s, lateral = lane.local_coordinates(agent.position)
        s, lateral = numeric(s), numeric(lateral)
        result['lane_current_local_longitudinal_lateral'] = [s, lateral]
        length = numeric(lane.length)
        result['lane_length_m'] = length
        for lookahead in (10, 25, 45):
            if 0 <= s + lookahead <= length:
                try:
                    result['lane_center_at_forward_dist'][str(lookahead)] = _lane_point(agent, s + lookahead)
                except Exception as exc:
                    result['lane_center_at_forward_dist'][str(lookahead)] = {'error': str(exc)}
            else:
                result['lane_center_at_forward_dist'][str(lookahead)] = None
    except Exception as exc:
        result['lane_probe_error'] = f'{type(exc).__name__}: {exc}'
    return result


def summarize_steering(trials):
    by_seed = defaultdict(dict)
    for r in trials:
        if not r['early_end'] and r['pulse_steps_completed'] == r['pulse_steps_requested']:
            by_seed[r['seed']][round(r['test_action'][0], 4)] = (r['yaw_delta_rad'], r['left_displacement_m'])
    result = []
    for seed, lookup in sorted(by_seed.items()):
        if 0.0 not in lookup:
            continue
        base_yaw, base_lat = lookup[0.0]
        for steer, (dyaw, lat) in sorted(lookup.items()):
            result.append({'seed': seed, 'steer': steer,
                           'yaw_change_minus_neutral_rad': dyaw - base_yaw,
                           'left_displacement_minus_neutral_m': lat - base_lat})
    plus = [r['yaw_change_minus_neutral_rad'] for r in result if r['steer'] == 0.25]
    minus = [r['yaw_change_minus_neutral_rad'] for r in result if r['steer'] == -0.25]
    plus_median, minus_median = median(plus), median(minus)
    plus_lat = median([r['left_displacement_minus_neutral_m'] for r in result if r['steer'] == 0.25])
    minus_lat = median([r['left_displacement_minus_neutral_m'] for r in result if r['steer'] == -0.25])
    # No automatic sign if missing trials, lack of movement, asymmetry, or weak effects.
    sign_confirmed = (plus_median is not None and minus_median is not None and
                      plus_median > 0.025 and minus_median < -0.025) or (
                      plus_median is not None and minus_median is not None and
                      plus_median < -0.025 and minus_median > 0.025)
    direction_consistent = bool(sign_confirmed and plus_lat is not None and minus_lat is not None and
                                abs(plus_lat) > 0.03 and abs(minus_lat) > 0.03 and
                                plus_lat * plus_median > 0 and minus_lat * minus_median > 0 and
                                len(plus) >= 2 and len(minus) >= 2)
    return {
        'neutral_corrected_per_seed': result,
        'plus_025_median_yaw_response_rad': plus_median,
        'minus_025_median_yaw_response_rad': minus_median,
        'plus_025_left_displacement_minus_neutral_m': plus_lat,
        'minus_025_left_displacement_minus_neutral_m': minus_lat,
        'left_right_direction_consistent_on_2plus_seeds': direction_consistent,
        'physical_direction_of_positive_metadrive_steer': ('left' if plus_lat > 0 else 'right') if direction_consistent else None,
        'candidate_carla_to_metadrive_steer_sign_not_applied': (-1 if plus_lat > 0 else 1) if direction_consistent else None,
        'bidirectional_sign_supported': bool(sign_confirmed),
        'metadrive_positive_steer_yaw_sign': (1 if plus_median > 0 else -1)
            if sign_confirmed else None,
        'important': 'Heading radians are simulator-coordinate positive; physical left/right convention still needs mapping to CARLA. No gain is inferred.',
    }


def summarize_longitudinal(trials, key):
    by_seed = defaultdict(dict)
    for r in trials:
        if not r['early_end'] and r['pulse_steps_completed'] == r['pulse_steps_requested']:
            by_seed[r['seed']][round(r['test_action'][1], 4)] = r['speed_delta_mps']
    rows = []
    for seed, vals in sorted(by_seed.items()):
        if 0 not in vals:
            continue
        for a2, delta in sorted(vals.items()):
            rows.append({'seed': seed, 'a2': a2,
                         'delta_speed_minus_neutral_mps': delta - vals[0]})
    summary = {}
    for a2 in sorted({row['a2'] for row in rows}):
        summary[str(a2)] = median([row['delta_speed_minus_neutral_mps'] for row in rows if row['a2'] == a2])
    return {key: rows, 'neutral_corrected_medians_mps': summary,
            'important': 'These are finite-horizon responses, not a calibrated CARLA-to-MetaDrive throttle/brake gain.'}


def navigation_pass(env, seeds, n_steps):
    samples, episodes = [], []
    for seed in seeds:
        env.reset(seed=seed)
        count, terminal = 0, False
        for step in range(n_steps):
            if step % 10 == 0:
                samples.append(nav_snapshot(env, step, seed))
            terminal, reason = step_command(env, 0.0, 0.38)
            count += 1
            if terminal:
                break
        episodes.append({'seed': seed, 'n_steps': count, 'early_end': terminal,
                         'end_info': reason if terminal else None})
    native_forward = [row['native_wp1_local_xy'][0] for row in samples]
    diff = []
    for row in samples:
        lp = row['lane_center_at_forward_dist'].get('25')
        if isinstance(lp, list) and len(lp) == 2:
            diff.append(abs(row['native_wp1_local_xy'][0] - lp[0]))
    return {
        'samples': samples, 'episodes': episodes,
        'navigation_summary': {'native_wp1_forward_median_m': median(native_forward),
                               'native_wp1_to_lane_25m_abs_forward_gap_median': median(diff),
                               'lane_25m_comparison_count': len(diff)},
        'warning': 'Current nav checkpoint and 25/45m lane center are NOT interchangeable on intersections; lane center lookahead is diagnostic only, not route-aware replacement.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('metadrive_calibration_v21.json'))
    parser.add_argument('--seeds', default='0,1', help='Comma-separated scenario seeds, default 0,1')
    parser.add_argument('--warmup-steps', type=int, default=34)
    parser.add_argument('--steer-steps', type=int, default=7)
    parser.add_argument('--speed-steps', type=int, default=12)
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        parser.error('Provide one or more distinct integer seeds')
    if min(args.warmup_steps, args.steer_steps, args.speed_steps) < 1:
        parser.error('All step counts must be positive')
    from metadrive import MetaDriveEnv
    config = dict(use_render=False, map='S', traffic_density=0.0,
                  num_scenarios=max(seeds)+1, log_level=40)
    env = MetaDriveEnv(config)
    steer_trials, accel_trials, brake_trials = [], [], []
    try:
        for seed in seeds:
            for steer in (0.0, -0.25, +0.25, -0.4, +0.4):
                steer_trials.append(run_trial(env, seed, args.warmup_steps, 0.52,
                                              (steer, 0.35), args.steer_steps))
            for a2 in (0.0, 0.2, 0.4, 0.7):
                accel_trials.append(run_trial(env, seed, 0, 0,
                                              (0.0, a2), args.speed_steps))
            for a2 in (0.0, -0.15, -0.4, -0.7):
                brake_trials.append(run_trial(env, seed, args.warmup_steps, 0.52,
                                              (0.0, a2), args.speed_steps))
        nav_report = navigation_pass(env, seeds, max(40, args.warmup_steps))
    finally:
        env.close()
    steering = summarize_steering(steer_trials)
    output = {
        'status': 'METADRIVE_ACTION_NAV_CALIBRATION_NOT_TRANSFER_CERTIFIED',
        'version': 'v2.1', 'config': config, 'seeds': seeds,
        'action_space_reference': 'MetaDrive action = [steer, throttle/brake], each in [-1,1]; a2>0 throttle, a2<0 brake.',
        'steering': {'trials': steer_trials, 'summary': steering},
        'acceleration': {'trials': accel_trials, 'summary': summarize_longitudinal(accel_trials, 'per_seed')},
        'braking': {'trials': brake_trials, 'summary': summarize_longitudinal(brake_trials, 'per_seed')},
        'navigation': nav_report,
        'provisional_recommendation': {
            'steering_sign_from_physics': steering['metadrive_positive_steer_yaw_sign'],
            'carla_to_metadrive_steering_multiplier': steering['candidate_carla_to_metadrive_steer_sign_not_applied'],
            'carla_to_metadrive_longitudinal_multiplier': None,
            'waypoint_adapter_patch': None,
            'reason': 'Require cross-simulator coordinate validation; a physical action test by itself does not determine dynamics-equivalent gains or route semantics.',
        },
        'limitations': [
            'No PDD data or learned policy used in this probe.',
            'Traffic is disabled and map is straight; results are for vehicle-level sign/response only.',
            'Longitudinal action effects are not a proof of cross-simulator equivalence.',
            'Lane-center lookahead is only valid within the current lane segment.',
            'No action scaling or waypoint transformation is automatically applied.',
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Steer +0.25 neutral-corrected yaw (rad):', steering['plus_025_median_yaw_response_rad'])
    print('Steer -0.25 neutral-corrected yaw (rad):', steering['minus_025_median_yaw_response_rad'])
    print('Steering sign supported:', steering['bidirectional_sign_supported'])
    print('Candidate CARLA->MetaDrive steer sign (NOT applied):', steering['candidate_carla_to_metadrive_steer_sign_not_applied'])
    print('Navigation samples:', len(nav_report['samples']))
    print('Saved:', args.out.resolve())
    print('NO learned action was used. Do not apply a sign/gain change before reviewing this report.')


if __name__ == '__main__':
    main()
