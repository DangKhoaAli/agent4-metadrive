"""V3.4: out-of-training-seed, speed-robustness benchmark for frozen Agent4 V3.3.

Research simulation only. Does not train or change the learned steering policy.
Only compares PDD baseline, manual residual, and learned residual on identical
MetaDrive seeds, maps, speed targets and unmodified safety terminations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parent.parent
DEPENDENCIES = [
    ('agent4_controller_v30', 'ab_closed_loop_v30.py'),
    ('agent4_controller_v31', 'profiles_v31.py'),
    ('agent4_learned_residual_v33', 'eval_student_v33.py'),
]
for directory, filename in DEPENDENCIES:
    required = ROOT / directory / filename
    if not required.is_file():
        raise RuntimeError(f'Missing sibling dependency: {required}')
    sys.path.insert(0, str(required.parent))

import ab_closed_loop_v30 as v30  # noqa: E402
import eval_student_v33 as v33  # noqa: E402
from profiles_v31 import episode_summary  # noqa: E402

ALLOWED_MAPS = ('S', 'C', 'X')
ALLOWED_PROFILES = ('baseline_3', 'lateral_3', 'student_v33')


def parse_seeds(value: str) -> list[int]:
    """Parse '32-35,40,42-44' into distinct nonnegative integers."""
    result = []
    for token in value.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            parts = token.split('-')
            if len(parts) != 2 or not all(x.isdigit() for x in parts):
                raise ValueError(f'Invalid seed range: {token}')
            lo, hi = (int(x) for x in parts)
            if lo > hi or hi - lo > 100:
                raise ValueError(f'Invalid or overly large seed range: {token}')
            result.extend(range(lo, hi + 1))
        else:
            if not token.isdigit():
                raise ValueError(f'Invalid seed: {token}')
            result.append(int(token))
    if not result or len(set(result)) != len(result):
        raise ValueError('Seed list must be nonempty and have no duplicates')
    return result


def parse_simple_csv(value: str, allowed: tuple[str, ...]) -> list[str]:
    result = [x.strip() for x in value.split(',') if x.strip()]
    if not result or len(set(result)) != len(result) or any(x not in allowed for x in result):
        raise ValueError(f'Choose distinct values from: {allowed}')
    return result


def parse_speeds(value: str) -> list[float]:
    try:
        speeds = [float(x.strip()) for x in value.split(',') if x.strip()]
    except ValueError as e:
        raise ValueError('Invalid speed list') from e
    # Upper bound is an experimental research constraint, not a safety guarantee.
    if not speeds or any(not math.isfinite(x) or x < 0.5 or x > 5.0 for x in speeds):
        raise ValueError('Speeds must be in the range 0.5 to 5.0 m/s')
    if len(set(speeds)) != len(speeds):
        raise ValueError('Duplicate speeds')
    return speeds


def attach_student_predictions(result: dict, predictions: list[dict]) -> dict:
    """Clarify V3.3 trace fields: model_steer_md was the *combined* steer.

    Predictions are aligned with the action steps because v30.pilot_episode
    predicts at most once per sequential env step, never skipping an applied step.
    """
    result['student_prediction_calls'] = len(predictions)
    result['learned_correction_active_steps'] = sum(
        abs(r['learned_residual']) > 1e-8 for r in predictions)
    result['learned_residual_max_abs'] = (
        max(abs(r['learned_residual']) for r in predictions) if predictions else None)
    result['student_predictions_aligned'] = (len(predictions) == result.get('n_applied_steps'))
    for item in result.get('trace', []):
        step = item.get('step')
        if isinstance(step, int) and 0 <= step < len(predictions):
            entry = predictions[step]
            item['steering_attribution_v34'] = {
                'original_pdd_steer_md': entry['original_pdd_steer_md'],
                'learned_residual_md': entry['learned_residual'],
                'student_proposed_steer_md': entry['student_proposed_steer_md'],
                'actually_applied_steer_md': float(item['action_applied'][0]),
            }
    return result


@contextmanager
def capture_v33_speed_and_predictions(speed: float, predictions: list[dict]):
    """Patch V3.3 entrypoint for one episode; always restore originals."""
    real_options = v33.options_for
    real_student_predict = v33.student_steer_md

    def options_at_speed(profile, **kwargs):
        opts = real_options(profile, **kwargs)
        opts.speed_limit_mps = speed
        return opts

    def record_student(model, row, names, max_steer, max_residual):
        pred = real_student_predict(model, row, names, max_steer, max_residual)
        predictions.append({
            'original_pdd_steer_md': float(pred['base']),
            'learned_residual': float(pred['learned_residual']),
            'student_proposed_steer_md': float(pred['steer']),
        })
        return pred

    v33.options_for = options_at_speed
    v33.student_steer_md = record_student
    try:
        yield
    finally:
        v33.options_for = real_options
        v33.student_steer_md = real_student_predict


def eval_one(env, map_name, seed, speed, profile, pdd, student, steps, trace_stride):
    predictions = []
    with capture_v33_speed_and_predictions(speed, predictions):
        result = v33.evaluate(env, seed, profile, pdd, student, steps, trace_stride)
    if profile == 'student_v33':
        attach_student_predictions(result, predictions)
    summary = episode_summary(result)
    if profile == 'student_v33':
        summary['student_prediction_calls'] = result['student_prediction_calls']
        summary['student_predictions_aligned'] = result['student_predictions_aligned']
        summary['learned_correction_active_steps'] = result['learned_correction_active_steps']
        summary['learned_residual_max_abs'] = result['learned_residual_max_abs']
        if not result['student_predictions_aligned']:
            result.setdefault('errors', []).append('Student prediction/action alignment mismatch')
            summary['error_count'] = len(result['errors'])
    return {
        'map': map_name, 'seed': seed, 'speed_mps': speed, 'profile': profile,
        'summary': summary, 'trace': result.get('trace', []),
        'errors': result.get('errors', []),
    }


def summary_rows(episodes: list[dict]) -> list[dict]:
    return [dict(map=e['map'], seed=e['seed'], speed_mps=e['speed_mps'],
                 profile=e['profile'], status=e['summary']['status'],
                 route_completion=e['summary']['route_completion'],
                 distance_m=e['summary']['travelled_approx_m'],
                 steps=e['summary']['n_applied_steps'],
                 max_lateral_m=e['summary']['max_abs_planned_lateral_right_m'],
                 max_heading_deg=(e['summary']['max_abs_heading_error_rad']*180/math.pi
                                  if e['summary']['max_abs_heading_error_rad'] is not None else None),
                 min_edge_clearance_m=e['summary']['min_estimated_edge_clearance_m'],
                 near_edge_warning_steps=e['summary']['near_edge_warning_steps'],
                 terminal_reasons=';'.join(e['summary']['terminal_reasons']),
                 error_count=e['summary']['error_count'],
                 student_prediction_calls=e['summary'].get('student_prediction_calls'),
                 student_predictions_aligned=e['summary'].get('student_predictions_aligned'),
                 learned_correction_active_steps=e['summary'].get('learned_correction_active_steps'),
                 learned_residual_max_abs=e['summary'].get('learned_residual_max_abs'))
            for e in episodes]


def group_scores(rows: list[dict]) -> list[dict]:
    keys = sorted({(r['speed_mps'],r['map'],r['profile']) for r in rows})
    return [{'speed_mps': speed, 'map': m, 'profile': profile,
             'success': sum(r['status'] == 'arrive_dest' for r in rows if
                            (r['speed_mps'],r['map'],r['profile'])==(speed,m,profile)),
             'total': sum((r['speed_mps'],r['map'],r['profile'])==(speed,m,profile)
                          for r in rows)}
            for speed,m,profile in keys]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--maps', default='C,X')
    p.add_argument('--seeds', default='32-35')
    p.add_argument('--speeds', default='3.0')
    p.add_argument('--profiles', default='baseline_3,lateral_3,student_v33')
    p.add_argument('--steps', type=int, default=1400)
    p.add_argument('--trace-stride', type=int, default=30)
    p.add_argument('--out', type=Path, default=Path('agent4_v34_robustness.json'))
    p.add_argument('--csv-out', type=Path, default=None)
    a = p.parse_args()
    try:
        maps = parse_simple_csv(a.maps, ALLOWED_MAPS)
        seeds = parse_seeds(a.seeds)
        speeds = parse_speeds(a.speeds)
        profiles = parse_simple_csv(a.profiles, ALLOWED_PROFILES)
        if a.steps < 50 or not 1 <= a.trace_stride <= 100:
            raise ValueError('Bad steps or trace-stride')
    except ValueError as e:
        p.error(str(e))

    pdd = joblib.load(a.model)
    student = joblib.load(a.student)
    if list(pdd.get('features', [])) != list(v30.FEATURES) or pdd.get('label_mode') != 'at_t':
        raise ValueError('Expected PDD V2 at_t feature order')
    v33.checked_student_bundle(student, v30.FEATURES, seeds)
    from metadrive.envs.metadrive_env import MetaDriveEnv

    report = {
        'status': 'V34_UNSEEN_SEED_AND_SPEED_STRESS_SIMULATION',
        'maps': maps, 'seeds': seeds, 'speeds_mps': speeds,
        'profiles': profiles, 'traffic_density': 0.0,
        'pdd_steering_model_frozen': True, 'student_steering_model_frozen': True,
        'learned_longitudinal_applied': False, 'idm_used_for_control': False,
        'training_or_tuning_in_this_run': False,
        'changes_from_v33': ['Selectable governor target speed',
                             'Unseen seed benchmark',
                             'Explicit raw PDD / learned residual steering attribution'],
        'warnings': [
            'Student V3.3 was trained at 3 m/s: other speeds are OUT-OF-DISTRIBUTION tests.',
            'Privileged MetaDrive road geometry; zero traffic; not validated on a real car.',
            'The longitudinal speed governor is rule-based, not a learned model.',
            'Only arrive_dest counts as success; lane guard status is NOT out_of_road.',
            'Do not use these evaluation seeds for later training or tuning.',
            'Estimated edge clearance is not the authoritative MetaDrive contact flag.',
            'V3.3 trace model_steer_md is the combined student result; V3.4 adds raw attribution.',
        ],
        'episodes': [], 'scores_by_speed_map_profile': [],
    }
    for map_name in maps:
        env = None
        try:
            env = MetaDriveEnv(dict(use_render=False, map=map_name,
                                    num_scenarios=max(seeds)+1,
                                    traffic_density=0.0,
                                    horizon=max(a.steps+10, 1000),log_level=40))
            for seed in seeds:
                for speed in speeds:
                    for profile in profiles:
                        entry = eval_one(env, map_name, seed, speed, profile,
                                         pdd, student, a.steps, a.trace_stride)
                        report['episodes'].append(entry)
                        s=entry['summary']
                        print(f'{map_name} seed={seed} speed={speed:g} profile={profile} '
                              f'status={s["status"]} distance={s["travelled_approx_m"]:.1f}m '
                              f'heading_max={s["max_abs_heading_error_rad"]*180/math.pi:.1f}deg')
        finally:
            if env is not None:
                env.close()

    rows = summary_rows(report['episodes'])
    report['scores_by_speed_map_profile'] = group_scores(rows)
    report['success_by_profile'] = {
        profile:{'success':sum(r['profile']==profile and r['status']=='arrive_dest' for r in rows),
                 'total':sum(r['profile']==profile for r in rows)}
        for profile in profiles
    }
    out_csv = a.csv_out or a.out.with_suffix('.csv')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    with out_csv.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('JSON:',a.out.resolve())
    print('CSV:',out_csv.resolve())
    print('Success:',report['success_by_profile'])
    print('IDM in control: NO; learned longitudinal: NO.')


if __name__ == '__main__':
    main()
