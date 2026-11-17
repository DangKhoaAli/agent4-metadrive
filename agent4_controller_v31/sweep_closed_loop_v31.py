"""V3.1 diagnostic sweep: same learned model, fixed experimental controls.

Requires sibling agent4_controller_v30 and its dependencies (V2.7, V2.9).
All experiments run in MetaDrive simulation; traffic disabled; no IDM.
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
V30 = ROOT / 'agent4_controller_v30' / 'ab_closed_loop_v30.py'
if not V30.is_file():
    raise RuntimeError(f'Missing V3.0 dependency: {V30}')
sys.path.insert(0, str(V30.parent))
from ab_closed_loop_v30 import pilot_episode, FEATURES  # noqa: E402
from profiles_v31 import (PROFILES, DEFAULT_PROFILES, select_profiles,
                          options_for, episode_summary)  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--maps', default='C')
    p.add_argument('--seeds', default='0')
    p.add_argument('--profiles', default=DEFAULT_PROFILES,
                   help=f'Comma-separated choices: {", ".join(PROFILES)}')
    p.add_argument('--steps', type=int, default=1400)
    p.add_argument('--trace-stride', type=int, default=10)
    p.add_argument('--tail-steps', type=int, default=35)
    p.add_argument('--out', type=Path, default=Path('agent4_v31_sweep_C.json'))
    a = p.parse_args()
    try:
        a.profiles = select_profiles(a.profiles)
        a.maps = [s.strip() for s in a.maps.split(',') if s.strip()]
        a.seeds = [int(s.strip()) for s in a.seeds.split(',') if s.strip()]
        if not a.maps or any(m not in {'S','C','X'} for m in a.maps):
            raise ValueError('maps must contain S, C or X')
        if not a.seeds or any(s < 0 for s in a.seeds):
            raise ValueError('seeds must be nonnegative integers')
        if len(set(a.maps)) != len(a.maps) or len(set(a.seeds)) != len(a.seeds):
            raise ValueError('maps and seeds must not repeat')
        options_for(PROFILES[a.profiles[0]], steps=a.steps,
                    trace_stride=a.trace_stride, tail_steps=a.tail_steps)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    return a


def main():
    a = parse_args()
    model = joblib.load(a.model)
    if list(model.get('features', [])) != FEATURES or model.get('label_mode') != 'at_t':
        raise ValueError('Expected V2 PDD model with at_t label and original FEATURES')
    from metadrive.envs.metadrive_env import MetaDriveEnv

    report = {
        'status':'V31_CONTROL_PARAMETER_AND_SPEED_SCREENING',
        'model_path':str(a.model), 'maps':a.maps,'seeds':a.seeds,
        'selected_profiles':a.profiles,
        'profiles':{n:asdict(PROFILES[n]) for n in a.profiles},
        'shared_settings':{
            'steps':a.steps, 'trace_stride':a.trace_stride,
            'tail_steps':a.tail_steps,
            'steer_gain':1.0, 'steer_limit':0.35,
            'max_steer_delta':0.12, 'guard_lane_offset_m':1.5,
            'guard_heading_rad':0.45, 'guard_road_margin_m':0.0,
            'warn_road_margin_m':0.35,'max_gap':2.0,
            'traffic_density':0.0,'idm_used':False,
            'learned_steering_applied':True,
            'learned_longitudinal_applied':False,
        },
        'episodes_by_map':[],
        'warnings':[
            'MetaDrive simulator testing only, not for physical vehicles.',
            'Residual corrections are NOT trained; success would not prove learned-only control.',
            'No exact equivalent action-gain calibration to CARLA is claimed.',
            'Same map/seed is a diagnostic comparison, not generalization evidence.',
            'Mode baseline_2 is required to isolate speed from residual improvement.',
            'Do not disable out-of-road continuous-line termination to claim success.',
            'Episode status and MetaDrive arrive_dest flag are authoritative; distance alone is not success.',
        ],
    }
    for map_name in a.maps:
        env = None
        episodes=[]
        try:
            env=MetaDriveEnv(dict(
                use_render=False, map=map_name, num_scenarios=max(a.seeds)+1,
                traffic_density=0.0, horizon=max(a.steps+10, 1000),log_level=40))
            for seed in a.seeds:
                for profile_name in a.profiles:
                    profile = PROFILES[profile_name]
                    opts = options_for(profile, steps=a.steps, trace_stride=a.trace_stride,
                                       tail_steps=a.tail_steps)
                    result=pilot_episode(env,seed,model,profile.mode,opts)
                    summary=episode_summary(result)
                    episodes.append({
                        'seed':seed,'profile':profile_name,
                        'settings':asdict(profile),'summary':summary,
                        'errors':result.get('errors',[]),
                        'trace':result.get('trace',[]),
                    })
                    rc=summary['route_completion']
                    route_str=(f'{rc*100:.2f}%' if isinstance(rc,(float,int))
                               and math.isfinite(rc) else 'n/a')
                    print(f'map={map_name} seed={seed} profile={profile_name} '
                          f'status={summary["status"]} steps={summary["n_applied_steps"]} '
                          f'distance~{summary["travelled_approx_m"]:.1f}m '
                          f'progress={route_str}')
                    # We still log errors, but never misclassify guard/step_limit as success.
        finally:
            if env is not None:
                env.close()
        report['episodes_by_map'].append({'map':map_name,'episodes':episodes})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Saved:',a.out.resolve())
    print('Only status=arrive_dest means success; all profiles retain learned steering, '
          'independent speed governor, strict MetaDrive line-contact termination, no IDM.')

if __name__=='__main__':
    main()
