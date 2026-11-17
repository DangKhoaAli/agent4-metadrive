"""V3.6.1 diagnostic ONLY: test whether V3.6's fast initial acceleration causes 4 m/s failures.

Three closed-loop arms on identical map/seed/speed, with the SAME frozen V3.5
steering and the SAME V3.6 learned longitudinal model:
  governor_v35              : original fixed governor
  learned_long_v36          : original learned throttle/brake, untouched
  learned_launch_cap_v361   : learned throttle/brake with first N commands capped
                             at +0.4 (braking predictions untouched)

The launch cap is a hand-designed diagnostic intervention, NOT a trained policy.
No retraining; no IDM used in evaluation; no traffic.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parent.parent
REQS = [
    ROOT / 'agent4_longitudinal_v36' / 'eval_longitudinal_v36.py',
    ROOT / 'agent4_multispeed_v35' / 'evaluate_multispeed_v35.py',
]
for path in REQS:
    if not path.is_file():
        raise RuntimeError(f'Missing required sibling dependency: {path}')
sys.path.insert(0, str(REQS[0].parent))
import eval_longitudinal_v36 as v36

PROFILES = ('governor_v35', 'learned_long_v36', 'learned_launch_cap_v361')
STATUS = 'V361_LAUNCH_ACCELERATION_ABLATION'


def validate_launch_params(steps: int, cap: float) -> None:
    if not isinstance(steps, int) or not 1 <= steps <= 200:
        raise ValueError('cap-steps must be an integer in 1..200')
    if not math.isfinite(cap) or not 0 < cap < 1:
        raise ValueError('launch cap must be in (0, 1)')


@contextmanager
def diagnostic_cap(steps: int = 40, cap: float = 0.4):
    """Patch V3.6's prediction output only during one episode.

    Counts actual prediction calls, not environment ticks. Restores function on
    success, exception or early termination. Does not alter the trained model.
    """
    validate_launch_params(steps, cap)
    actual_predict = v36.predict_action
    stats = {'prediction_calls': 0, 'cap_active_calls': 0,
             'capped_positive_calls': 0, 'max_raw_launch_action': None,
             'launch_cap': float(cap), 'launch_cap_steps': int(steps)}

    def intercepted(bundle: Any, row: Any) -> float:
        raw = float(actual_predict(bundle, row))
        index = stats['prediction_calls']
        stats['prediction_calls'] += 1
        if index < steps:
            stats['cap_active_calls'] += 1
            high = stats['max_raw_launch_action']
            stats['max_raw_launch_action'] = raw if high is None else max(high, raw)
            if raw > cap:
                stats['capped_positive_calls'] += 1
                return float(cap)
        return raw

    v36.predict_action = intercepted
    try:
        yield stats
    finally:
        v36.predict_action = actual_predict


def run_one(env, map_name: str, seed: int, speed: float, profile: str,
            pdd, steering, longitudinal, steps: int, stride: int,
            launch_steps: int, launch_cap: float) -> dict:
    if profile not in PROFILES:
        raise ValueError(f'Bad profile: {profile}')

    if profile == 'learned_launch_cap_v361':
        with diagnostic_cap(launch_steps, launch_cap) as stats:
            episode = v36.one(env, map_name, seed, speed, 'learned_long_v36',
                              pdd, steering, longitudinal, steps, stride)
        if stats['prediction_calls'] != episode['summary']['n_applied_steps']:
            episode['errors'].append('Launch cap count did not match applied steps')
        episode['summary']['launch_diagnostic'] = dict(stats)
    else:
        episode = v36.one(env, map_name, seed, speed, profile,
                          pdd, steering, longitudinal, steps, stride)
        episode['summary']['launch_diagnostic'] = None
    episode['profile'] = profile
    for event in episode.get('trace', []):
        attrib = event.get('longitudinal_attribution_v36')
        if isinstance(attrib, dict):
            attrib['mode'] = profile
    episode['summary']['error_count'] = len(episode['errors'])
    # Speed checkpoints are only recorded if the corresponding trace sample exists.
    episode['summary']['speed_checkpoints_mps'] = {
        str(step): t['speed_mps_before'] for t in episode.get('trace', [])
        if (step := t.get('step')) in (0, 10, 20, 30, 40, 60)
    }
    return episode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pdd', type=Path, required=True)
    ap.add_argument('--steering', type=Path, required=True)
    ap.add_argument('--longitudinal', type=Path, required=True)
    ap.add_argument('--maps', default='C,X')
    ap.add_argument('--seeds', default='76,77')
    ap.add_argument('--speeds', default='4')
    ap.add_argument('--profiles', default=','.join(PROFILES))
    ap.add_argument('--steps', type=int, default=1800)
    ap.add_argument('--trace-stride', type=int, default=10)
    ap.add_argument('--launch-steps', type=int, default=40)
    ap.add_argument('--launch-cap', type=float, default=0.4)
    ap.add_argument('--out', type=Path, default=Path('agent4_v361_launch_ablation.json'))
    args = ap.parse_args(argv)
    try:
        maps = v36.parse_simple_csv(args.maps, v36.MAPS)
        seeds = v36.parse_seeds(args.seeds)
        speeds = v36.parse_speeds(args.speeds)
        profiles = v36.parse_simple_csv(args.profiles, PROFILES)
        validate_launch_params(args.launch_steps, args.launch_cap)
        if not 100 <= args.steps <= 5000 or not 1 <= args.trace_stride <= 100:
            raise ValueError('Invalid steps or trace-stride')
    except ValueError as exc:
        ap.error(str(exc))

    pdd = joblib.load(args.pdd)
    steering = joblib.load(args.steering)
    longitudinal = joblib.load(args.longitudinal)
    if pdd.get('label_mode') != 'at_t' or len(pdd.get('features', [])) != 15:
        raise ValueError('Expected PDD V2 at_t with 15 features')
    v36.check_student(steering, pdd['features'], seeds,
                      'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    if longitudinal.get('version') != v36.VERSION:
        raise ValueError('Wrong V3.6 longitudinal model version')
    if longitudinal.get('feature_order') != v36.names(pdd['features']):
        raise ValueError('V3.6 longitudinal model feature ordering mismatch')
    if longitudinal.get('pdd_features') != pdd['features']:
        raise ValueError('PDD feature contract mismatch')
    v36.check_eval_seeds(longitudinal, seeds)
    if not set(speeds) <= set(longitudinal['speed_targets']):
        raise ValueError('Requested speeds not included in V3.6 training')

    from metadrive.envs.metadrive_env import MetaDriveEnv
    report = {
        'status': STATUS,
        'maps': maps, 'seeds': seeds, 'speeds_mps': speeds,
        'profiles': profiles,
        'launch_cap_steps': args.launch_steps,
        'launch_cap_positive': args.launch_cap,
        'same_frozen_v35_steering': True,
        'same_trained_v36_longitudinal_model_for_both_learned_arms': True,
        'idm_used_for_control': False, 'traffic_density': 0.0,
        'training_in_this_run': False,
        'warnings': [
            'Launch cap is a rule-based diagnostic intervention, NOT part of learned V3.6 model.',
            'Seeds 76 and 77 have already been tested; findings are diagnostic, not heldout generalization.',
            'Even if capped launch succeeds, this does not prove the sole cause of failures.',
            'Only arrive_dest counts as success. Road edge clearance is approximate.',
            'MetaDrive simulation only; no traffic and privileged geometry; not physical safety evidence.'
        ],
        'episodes': [], 'success_by_profile': {}, 'scores_by_speed_map_profile': []
    }
    for map_name in maps:
        env = None
        try:
            env = MetaDriveEnv(dict(use_render=False, map=map_name,
                                    num_scenarios=max(seeds) + 1, traffic_density=0.0,
                                    horizon=max(args.steps + 10, 1000), log_level=40))
            for seed in seeds:
                for speed in speeds:
                    for profile in profiles:
                        item = run_one(env, map_name, seed, speed, profile, pdd,
                                       steering, longitudinal, args.steps,
                                       args.trace_stride, args.launch_steps, args.launch_cap)
                        report['episodes'].append(item)
                        summary = item['summary']
                        print(f'{map_name} seed={seed} speed={speed:g} {profile} '
                              f'-> {summary["status"]}; distance={summary["travelled_approx_m"]:.1f}m '
                              f'speed_step20={summary["speed_checkpoints_mps"].get("20")}', flush=True)
        finally:
            if env is not None:
                env.close()
    rows = v36.summary_rows(report['episodes'])
    report['scores_by_speed_map_profile'] = v36.group_scores(rows)
    report['success_by_profile'] = {
        p: {'success': sum(e['profile'] == p and e['summary']['status'] == 'arrive_dest'
                           for e in report['episodes']),
            'total': sum(e['profile'] == p for e in report['episodes'])}
        for p in profiles
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                        encoding='utf-8')
    print('Results:', args.out.resolve())
    print('Success:', report['success_by_profile'])


if __name__ == '__main__':
    main()
