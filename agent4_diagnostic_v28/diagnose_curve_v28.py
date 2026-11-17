"""Agent 4 V2.8: reproducibility diagnostic for V2.7 guarded curve failure.

Same steering model, steering mapping, governor and guards as V2.7.
Adds full-step route, lane, heading and MetaDrive termination logging.
No IDM fallback, no gain adjustment, no autonomous longitudinal model.
Simulation-only. Requires existing sibling folders agent4_calibration_v27 etc.
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
V27 = ROOT / 'agent4_calibration_v27'
if not (V27 / 'guarded_pilot_v27.py').is_file():
    raise RuntimeError(f'Missing dependency: {V27 / "guarded_pilot_v27.py"}')
sys.path.insert(0, str(V27))

from guarded_pilot_v27 import (  # noqa: E402
    FEATURES,
    adaptive_horizons,
    bounded_steering,
    feature_for_model,
    governed_speed_action,
    lane_offset,
    predict_steering,
    choose_lane_chain,
    collect_forward_segments,
)

FLAG_NAMES = (
    'arrive_dest', 'out_of_road', 'out_of_route', 'crash', 'crash_vehicle',
    'crash_object', 'crash_sidewalk', 'on_lane', 'on_broken_line',
    'on_yellow_continuous_line', 'on_white_continuous_line',
    'route_completion', 'episode_length',
)


def float_or_none(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def angle_delta_radians(a, b):
    """Return a-b normalized to [-pi, pi]."""
    return math.atan2(math.sin(a-b), math.cos(a-b))


def compact_lane_state(vehicle):
    """A nearest-lane projection, not a verified drivable-road margin."""
    lane = getattr(vehicle, 'lane', None)
    record = {
        'lane_index': repr(getattr(vehicle, 'lane_index', None)),
        'lane_class': type(lane).__name__ if lane is not None else None,
        'lane_width_m': None,
        'lane_length_m': None,
        'lane_longitudinal_m': None,
        'lane_lateral_m': None,
        'lane_heading_rad': None,
        'heading_error_rad': None,
        'lane_longitudinal_in_range': None,
        'vehicle_heading_rad': float_or_none(getattr(vehicle, 'heading_theta', None)),
        'vehicle_position_xy': [float(vehicle.position[0]), float(vehicle.position[1])],
    }
    if lane is None:
        return record
    try:
        lon, lat = lane.local_coordinates(vehicle.position)
        lon, lat = float(lon), float(lat)
        length = float(lane.length)
        record['lane_width_m'] = float_or_none(lane.width)
        record['lane_length_m'] = float_or_none(length)
        record['lane_longitudinal_m'] = float_or_none(lon)
        record['lane_lateral_m'] = float_or_none(lat)
        record['lane_longitudinal_in_range'] = bool(-0.05 <= lon <= length + 0.05)
        # Compute tangent from actual lane geometry rather than assuming a global sign.
        s0 = min(max(lon-0.2, 0.0), max(length-1e-4, 0.0))
        s1 = min(max(lon+0.2, s0+1e-4), length)
        if s1 > s0:
            p0 = lane.position(s0, 0)
            p1 = lane.position(s1, 0)
            heading = math.atan2(float(p1[1]-p0[1]), float(p1[0]-p0[0]))
            record['lane_heading_rad'] = float_or_none(heading)
            vh = record['vehicle_heading_rad']
            if vh is not None:
                record['heading_error_rad'] = float_or_none(angle_delta_radians(vh, heading))
    except (AttributeError, TypeError, ValueError, RuntimeError, IndexError, KeyError):
        pass
    return record


def planned_lane_state(vehicle, gap=2.0):
    result = {
        'planned_lane_index': None,
        'planned_lane_lateral_m': None,
        'planned_lane_longitudinal_m': None,
        'planned_lane_width_m': None,
        'planned_lane_heading_error_rad': None,
        'planned_road_nodes': None,
        'lane_chain_gap_count': None,
        'error': None,
    }
    try:
        segments = collect_forward_segments(vehicle.navigation, vehicle)
        chain, gaps = choose_lane_chain(segments, vehicle.position, max_gap_m=gap)
        current = chain[0]
        lane = current['lane']
        lon, lat = lane.local_coordinates(vehicle.position)
        result['planned_lane_index'] = current['lane_index']
        result['planned_lane_lateral_m'] = float_or_none(lat)
        result['planned_lane_longitudinal_m'] = float_or_none(lon)
        result['planned_lane_width_m'] = float_or_none(lane.width)
        result['planned_road_nodes'] = [str(n) for n in current['nodes']]
        result['lane_chain_gap_count'] = len(gaps)
        length = float(lane.length)
        s0 = min(max(float(lon)-0.2,0.0), max(length-1.e-4,0.0))
        s1 = min(max(float(lon)+0.2,s0+1.e-4),length)
        if s1 > s0:
            p0=lane.position(s0,0); p1=lane.position(s1,0)
            heading=math.atan2(float(p1[1]-p0[1]), float(p1[0]-p0[0]))
            vehheading=float_or_none(getattr(vehicle,'heading_theta',None))
            if vehheading is not None:
                result['planned_lane_heading_error_rad'] = angle_delta_radians(vehheading, heading)
    except Exception as e:  # diagnostic logging must not alter the closed-loop controller
        result['error'] = f'{type(e).__name__}: {e}'
    return result


def safe_flags(info):
    flags = {}
    for k in FLAG_NAMES:
        if k not in info:
            continue
        value = info[k]
        if isinstance(value, (bool, np.bool_)):
            flags[k] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            flags[k] = float_or_none(value)
        else:
            flags[k] = repr(value)[:120]
    return flags


def diagnostic_episode(env, seed, steps, model, speed_limit, gain=1.0,
                       steer_limit=0.35, max_delta=0.12, guard_offset=1.5, max_gap=2.0):
    events = []
    errors = []
    status = 'step_limit'
    ended = None
    travel = 0.0
    previous = 0.0
    try:
        env.reset(seed=seed)
        for t in range(steps):
            vehicle = env.agent
            before = compact_lane_state(vehicle)
            before_plan = planned_lane_state(vehicle, max_gap)
            offset, width = lane_offset(vehicle)
            if offset is not None and abs(offset) > guard_offset:
                status = 'lane_departure_guard'
                events.append({'step': t, 'guard_event': status,
                               'before': before, 'before_plan': before_plan})
                break
            try:
                features, route = feature_for_model(env, max_gap)
                steer_carla = predict_steering(model, features)
                steer = bounded_steering(steer_carla, gain, steer_limit, previous, max_delta)
                longitudinal, target = governed_speed_action(features['speed_mps'], speed_limit, route['remaining_m'])
            except Exception as e:
                status = 'observation_or_model_guard'
                errors.append(f'{type(e).__name__}: {e}')
                events.append({'step': t, 'guard_event': status,
                               'before': before, 'before_plan': before_plan})
                break
            action = [float(steer), float(longitudinal)]
            before_xy = np.asarray(vehicle.position[:2], dtype=float)
            try:
                _, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                status = 'env_step_error'
                errors.append(f'{type(e).__name__}: {e}')
                break
            after = compact_lane_state(env.agent)
            after_plan = planned_lane_state(env.agent, max_gap)
            after_xy = np.asarray(env.agent.position[:2], dtype=float)
            move_m = float(np.linalg.norm(after_xy - before_xy))
            travel += move_m
            events.append({
                'step': t,
                'speed_mps_before': float(features['speed_mps']),
                'target_speed_mps': target,
                'model_steer_carla': float(steer_carla),
                'model_steer_md_sign_only': -float(steer_carla),
                'action_applied': action,
                'remaining_route_m': route['remaining_m'],
                'lookahead_m': route['effective_lookahead_m'],
                'lookahead_adapted': bool(route['adapted']),
                'model_wp1_right_m': float(features['wp1_right_m']),
                'model_wp2_right_m': float(features['wp2_right_m']),
                'before': before,
                'before_plan': before_plan,
                'after': after,
                'after_plan': after_plan,
                'move_m': move_m,
                'reward': float(reward),
                'sim_info': safe_flags(info),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
            })
            previous = steer
            if terminated or truncated:
                ended = {'terminated': bool(terminated), 'truncated': bool(truncated), **safe_flags(info)}
                if bool(info.get('arrive_dest', False)):
                    status = 'arrive_dest'
                elif bool(info.get('out_of_road', False)):
                    status = 'out_of_road'
                elif bool(info.get('crash', False)):
                    status = 'crash'
                else:
                    status = 'terminated_or_truncated'
                break
    except Exception as e:
        status = 'episode_error'
        errors.append(f'{type(e).__name__}: {e}')
    pre_offsets = [abs(x['before']['lane_lateral_m']) for x in events
                   if 'before' in x and x['before']['lane_lateral_m'] is not None]
    plan_offsets = [abs(x['before_plan']['planned_lane_lateral_m']) for x in events
                    if 'before_plan' in x and x['before_plan']['planned_lane_lateral_m'] is not None]
    lane_idx_changes = sum(events[i]['before']['lane_index'] != events[i-1]['before']['lane_index']
                           for i in range(1, len(events))
                           if 'before' in events[i] and 'before' in events[i-1])
    return {
        'seed': seed, 'status': status, 'n_applied_steps': sum('action_applied' in x for x in events),
        'travelled_approx_m': travel,
        'max_abs_current_lane_offset_before_m': max(pre_offsets, default=None),
        'max_abs_planned_lane_offset_before_m': max(plan_offsets, default=None),
        'current_lane_id_changes': lane_idx_changes,
        'end_info': ended,
        'errors': errors,
        'trace_every_step': events,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--maps', default='C')
    p.add_argument('--seeds', default='0')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--speed-limit-mps', type=float, default=3.0)
    p.add_argument('--steer-gain', type=float, default=1.0)
    p.add_argument('--steer-limit', type=float, default=0.35)
    p.add_argument('--max-steer-delta', type=float, default=0.12)
    p.add_argument('--guard-lane-offset-m', type=float, default=1.5)
    p.add_argument('--max-gap', type=float, default=2.0)
    p.add_argument('--out', type=Path, default=Path('agent4_v28_diagnostic_C.json'))
    a = p.parse_args()
    names = [v.strip() for v in a.maps.split(',') if v.strip()]
    seeds = [int(v.strip()) for v in a.seeds.split(',') if v.strip()]
    if not names or not seeds or min(seeds) < 0 or not 0 < a.speed_limit_mps <= 6 or not 0 < a.steer_gain <= 3:
        p.error('Invalid maps, seeds or speed/gain')
    if not (0 < a.steer_limit <= 0.7 and 0 < a.max_steer_delta <= 0.5 and
            0 < a.guard_lane_offset_m <= 2 and 0 < a.max_gap <= 5 and a.steps >= 5):
        p.error('Invalid steering/guard settings')
    model = joblib.load(a.model)
    if list(model.get('features', [])) != FEATURES or model.get('label_mode') != 'at_t':
        raise ValueError('Model schema mismatch with PDD V2')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report = {
        'status': 'V28_CURVE_DIAGNOSTIC_IDENTICAL_V27_CONTROL_WITH_EXTRA_LOGGING',
        'note': 'Same learned steering and independent speed governor as V2.7; no policy change.',
        'maps': names,
        'seeds': seeds,
        'settings': {
            'steps': a.steps, 'speed_limit_mps': a.speed_limit_mps,
            'steer_gain': a.steer_gain, 'steer_limit': a.steer_limit,
            'max_steer_delta': a.max_steer_delta,
            'guard_lane_offset_m': a.guard_lane_offset_m,
            'max_gap': a.max_gap,
            'traffic_density': 0.0,
        },
        'learned_steering_applied': True,
        'learned_longitudinal_applied': False,
        'idm_used': False,
        'episodes_by_map': [],
        'warnings': [
            'The lane projection is NOT the same as MetaDrive out_of_road geometry.',
            'Planned lane comes from continuity heuristic, not guaranteed route-turn correctness.',
            'This report is a diagnostic for simulation; not road-safety validation.',
        ],
    }
    for map_name in names:
        episodes = []
        env = None
        try:
            env = MetaDriveEnv(dict(use_render=False, map=map_name,
                                    num_scenarios=max(seeds)+1, traffic_density=0.0,
                                    horizon=max(a.steps+10, 1000), log_level=40))
            for seed in seeds:
                result = diagnostic_episode(env, seed, a.steps, model, a.speed_limit_mps,
                                            a.steer_gain, a.steer_limit, a.max_steer_delta,
                                            a.guard_lane_offset_m, a.max_gap)
                episodes.append(result)
                print(f'map={map_name} seed={seed} status={result["status"]} '
                      f'steps={result["n_applied_steps"]} distance~{result["travelled_approx_m"]:.1f}m '
                      f'lane_id_changes={result["current_lane_id_changes"]}')
        finally:
            if env is not None:
                env.close()
        report['episodes_by_map'].append({'map': map_name, 'episodes': episodes})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('Saved:', a.out.resolve())
    print('V2.7 control reproduced. Model steering: YES; model longitudinal: NO; IDM: NO.')

if __name__ == '__main__':
    main()
