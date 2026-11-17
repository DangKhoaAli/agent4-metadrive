"""V3.11B: frozen V3.10 lateral generalization diagnostic on MetaDrive C/X.

Uses EXACT V3.10 closed-loop adapter, PDD and rule-lateral reference.
Models and governor are never retrained or modified. Simulation only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parent.parent
V310_PATH = ROOT / 'agent4_highspeed_training_v310' / 'eval_v310.py'
if not V310_PATH.is_file():
    raise RuntimeError(f'Missing V3.10 dependency {V310_PATH}; extract both folders side by side')
sys.path.insert(0, str(V310_PATH.parent))
import eval_v310 as v310
from common_v310 import check_pdd, check_bundle, parse_seed_spec, parse_values

VERSION = 'V311B_FROZEN_V310_CURVED_GENERALIZATION'
MAPS = ('C', 'X', 'S')
PROFILES = ('pdd_only', 'rule_lateral', 'learned_v310')
STEP_SECONDS = .10
COLUMNS = ('map', 'seed', 'speed_mps', 'profile', 'status', 'arrive_dest',
           'n_applied_steps', 'time_to_goal_s', 'max_speed_mps',
           'max_abs_planned_lateral_right_m', 'min_estimated_edge_clearance_m',
           'near_edge_warning_steps', 'error_count', 'student_predictions_aligned',
           'model_predictions_valid', 'out_of_v310_training_speed')


def valid_speeds(spec):
    try:
        seq = [float(x.strip()) for x in spec.split(',') if x.strip()]
    except ValueError as exc:
        raise ValueError('Speeds must be comma-separated numbers in [1, 4] m/s') from exc
    if not seq or len(seq) != len(set(seq)) or any(not math.isfinite(x) or not 1. <= x <= 4. for x in seq):
        raise ValueError('Speeds must be distinct finite values in [1, 4] m/s')
    return seq


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def evaluation_config(args, maps, seeds, speeds, profiles):
    return {'maps': maps, 'seeds': seeds, 'speeds_mps': speeds, 'profiles': profiles,
            'steps': args.steps, 'trace_stride': args.trace_stride,
            'pdd_sha256': digest(args.pdd), 'steering_sha256': digest(args.steering),
            'traffic_density': 0., 'physics_world_step_size': .02, 'decision_repeat': 5}


def keys(episode):
    return (episode['map'], episode['seed'], episode['speed_mps'], episode['profile'])


def episode_is_success(e):
    s = e['summary']
    if s.get('status') != 'arrive_dest' or s.get('error_count', 0):
        return False
    if e['profile'] == 'learned_v310' and not s.get('model_predictions_valid', False):
        return False
    return True


def enrich_episode(e):
    s = e['summary']
    learned = e['profile'] == 'learned_v310'
    n = s.get('n_applied_steps', 0)
    s['out_of_v310_training_speed'] = bool(e['speed_mps'] < 4.)
    s['curved_map_out_of_v310_training_distribution'] = e['map'] in ('C', 'X')
    # V3.10 captures one student prediction per *applied* action and records
    # sparse trace attributions, but DOES NOT produce 'measured_steps_aligned'.
    # An absent optional metric must not invalidate otherwise verified episodes.
    # Explicit False, if supplied by a future collector, is still a veto.
    trace = e.get('trace', [])
    attribution_ok = bool(trace) and all(
        isinstance(t.get('v310_attribution'), dict) and
        math.isfinite(float(t['v310_attribution']['model_proposed_action'])) and
        math.isfinite(float(t['v310_attribution']['actual_applied_action'])) and
        abs(float(t['v310_attribution']['model_proposed_action']) -
            float(t['v310_attribution']['actual_applied_action'])) <= 1e-8
        for t in trace
    ) if learned else True
    s['model_predictions_valid'] = bool(not learned or (
        n > 0 and s.get('learned_steering_applied') is True and
        s.get('student_predictions_aligned') is True and
        s.get('student_prediction_calls') == n and
        s.get('teacher_fallback_calls') == 0 and
        s.get('measured_steps_aligned', True) is not False and
        s.get('error_count', 0) == 0 and not e.get('errors') and
        attribution_ok
    ))
    s['confirmed_success'] = episode_is_success(e)
    return e


def stats_for(group):
    success = [e for e in group if episode_is_success(e)]
    times = [e['summary']['time_to_goal_s'] for e in success
             if e['summary'].get('time_to_goal_s') is not None]
    lateral = [e['summary']['max_abs_planned_lateral_right_m'] for e in group
               if e['summary'].get('max_abs_planned_lateral_right_m') is not None]
    margin = [e['summary']['min_estimated_edge_clearance_m'] for e in group
              if e['summary'].get('min_estimated_edge_clearance_m') is not None]
    return {'total': len(group), 'arrived': sum(e['summary'].get('status') == 'arrive_dest' for e in group),
            'success': len(success), 'near_edge_warning_steps': sum(int(e['summary'].get('near_edge_warning_steps') or 0) for e in group),
            'max_lateral_m': max(lateral, default=None), 'min_estimated_edge_clearance_m': min(margin, default=None),
            'mean_max_lateral_m': sum(lateral) / len(lateral) if lateral else None,
            'mean_success_time_s': sum(times) / len(times) if times else None,
            'failures': [{'seed': e['seed'], 'status': e['summary'].get('status'),
                          'errors': e.get('errors', [])[:2]}
                         for e in group if not episode_is_success(e)]}


def aggregate(episodes):
    by = defaultdict(list)
    for e in episodes:
        by[(e['map'], e['speed_mps'], e['profile'])].append(e)
    return [dict(map=k[0], speed_mps=k[1], profile=k[2], **stats_for(g))
            for k, g in sorted(by.items())]


def pairwise(episodes):
    # Paired comparisons use identical map, seed and speed, never compare independent episodes.
    indexed = defaultdict(dict)
    for e in episodes:
        indexed[(e['map'], e['seed'], e['speed_mps'])][e['profile']] = e
    result = []
    for (m, seed, speed), profiles in sorted(indexed.items()):
        if 'learned_v310' not in profiles:
            continue
        student = profiles['learned_v310']
        ss = student['summary']
        for baseline in ('rule_lateral', 'pdd_only'):
            if baseline not in profiles:
                continue
            other = profiles[baseline]
            bs = other['summary']
            lateral_student = ss.get('max_abs_planned_lateral_right_m')
            lateral_base = bs.get('max_abs_planned_lateral_right_m')
            margin_student = ss.get('min_estimated_edge_clearance_m')
            margin_base = bs.get('min_estimated_edge_clearance_m')
            result.append(dict(map=m, seed=seed, speed_mps=speed, baseline=baseline,
                learner_success=episode_is_success(student), baseline_success=episode_is_success(other),
                learner_status=ss.get('status'), baseline_status=bs.get('status'),
                learner_minus_baseline_max_lateral_m=(lateral_student-lateral_base
                    if lateral_student is not None and lateral_base is not None else None),
                learner_minus_baseline_est_edge_clearance_m=(margin_student-margin_base
                    if margin_student is not None and margin_base is not None else None)))
    return result


def default_report(config):
    return {'version': VERSION, 'config': config, 'frozen_pdd': True,
            'frozen_v310_steering': True, 'teacher_fallback_allowed': False,
            'learned_longitudinal_used': False, 'model_training_performed': False,
            'warnings': [
                'Diagnostic evaluation, not a new model and not an untouched final holdout.',
                'V3.10 was trained on straight S/SSSSS at 4,6,8,10 m/s; C/X curves are out of distribution.',
                'At 2 and 3 m/s, vehicle speed is also out of V3.10 training target range.',
                'Only arrive_dest with aligned learned predictions and no errors counts as confirmed success.',
                'lane_departure_guard is a diagnostic guard; out_of_road is simulator road termination.',
                'Minimum estimated edge clearance is a proxy, not guaranteed physical clearance.',
                'PDD is a pretrained model; learned_v310 uses a learned steering residual; longitudinal is rule-based.',
                'Privileged navigation geometry, no traffic, MetaDrive simulation ONLY, not safe for a physical car.',
            ],
            'episodes': [], 'group_scores': [], 'paired_comparisons': []}


def refresh(report):
    report['group_scores'] = aggregate(report['episodes'])
    report['paired_comparisons'] = pairwise(report['episodes'])
    return report


def atomic_json(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(target)


def export_csv(report, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=COLUMNS)
        writer.writeheader()
        for e in report['episodes']:
            s = e['summary']
            out = {'map': e['map'], 'seed': e['seed'], 'speed_mps': e['speed_mps'],
                   'profile': e['profile'], 'status': s.get('status'),
                   'arrive_dest': int(episode_is_success(e))}
            out.update({field: s.get(field) for field in COLUMNS if field not in out})
            writer.writerow(out)
    temp.replace(target)


def init_report(path, config, resume, overwrite):
    if path.exists() and resume:
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('version') != VERSION or report.get('config') != config:
            raise ValueError('Existing report version/config/model hashes differ; use another --out path')
        existing = [keys(x) for x in report.get('episodes', [])]
        if len(set(existing)) != len(existing):
            raise ValueError('Existing report has duplicate episode keys')
        return report
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path}: already exists. Use --resume or --overwrite explicitly')
    return default_report(config)


def validate_args(args):
    maps = parse_values(args.maps, MAPS)
    seeds = parse_seed_spec(args.seeds)
    speeds = valid_speeds(args.speeds)
    profiles = parse_values(args.profiles, PROFILES)
    if min(seeds) < 158:
        raise ValueError('V3.11B diagnostic seeds must be >=158, distinct from prior V3.10/V3.11 evaluations')
    if not 100 <= args.steps <= 6000 or not 1 <= args.trace_stride <= 100:
        raise ValueError('steps must be 100..6000, trace stride 1..100')
    if args.resume and args.overwrite:
        raise ValueError('Do not combine --resume and --overwrite')
    return maps, seeds, speeds, profiles


def run(args):
    maps, seeds, speeds, profiles = validate_args(args)
    if not args.pdd.is_file() or not args.steering.is_file():
        raise FileNotFoundError('PDD / V3.10 .joblib missing; check --pdd and --steering paths')
    config = evaluation_config(args, maps, seeds, speeds, profiles)
    report = init_report(args.out, config, args.resume, args.overwrite)
    pdd = joblib.load(args.pdd)
    pdd_names = check_pdd(pdd)
    model = joblib.load(args.steering)
    check_bundle(model, pdd_names, seeds)
    from metadrive.envs.metadrive_env import MetaDriveEnv
    finished = {keys(x) for x in report['episodes']}
    try:
        for m in maps:
            requested = [(seed, speed, profile) for seed in seeds for speed in speeds
                         for profile in profiles if (m, seed, speed, profile) not in finished]
            if not requested:
                continue
            env = MetaDriveEnv(dict(use_render=False, map=m, num_scenarios=max(seeds) + 1,
                        traffic_density=0., horizon=max(1000, args.steps + 10), log_level=40,
                        physics_world_step_size=.02, decision_repeat=5))
            try:
                for seed, speed, profile in requested:
                    e = v310.run_one(env, m, seed, speed, profile, pdd,
                                     model, pdd_names, args.steps, args.trace_stride)
                    enrich_episode(e)
                    report['episodes'].append(e)
                    finished.add(keys(e))
                    refresh(report)
                    atomic_json(args.out, report)
                    export_csv(report, args.out.with_suffix('.csv'))
                    s = e['summary']
                    lateral = s.get('max_abs_planned_lateral_right_m')
                    lateral_text = f'{lateral:.3f}m' if lateral is not None else 'NA'
                    print(f'{len(report["episodes"]):3d} {m:2s} seed={seed} '
                          f'v={speed:g} {profile:13s} {s["status"]:28s} '
                          f'confirmed={int(episode_is_success(e))} lateral={lateral_text} '
                          f'edge_warn={s.get("near_edge_warning_steps")}', flush=True)
            finally:
                env.close()
    except KeyboardInterrupt:
        print('Interrupted. Previously completed episodes remain saved; run with --resume.', file=sys.stderr)
        raise
    refresh(report)
    atomic_json(args.out, report)
    export_csv(report, args.out.with_suffix('.csv'))
    print(f'Finished {len(report["episodes"])} episodes. JSON: {args.out.resolve()}')
    print(f'CSV: {args.out.with_suffix(".csv").resolve()}')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd', required=True, type=Path)
    p.add_argument('--steering', required=True, type=Path)
    p.add_argument('--maps', default='C,X')
    p.add_argument('--seeds', default='158-161')
    p.add_argument('--speeds', default='2,3,4')
    p.add_argument('--profiles', default=','.join(PROFILES))
    p.add_argument('--steps', default=2500, type=int)
    p.add_argument('--trace-stride', default=25, type=int)
    p.add_argument('--out', default=Path('agent4_v311b_curves.json'), type=Path)
    p.add_argument('--resume', action='store_true', help='Skip completed episodes only if config and model hashes match')
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace existing report')
    return run(p.parse_args(argv))


if __name__ == '__main__':
    main()
