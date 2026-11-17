"""V3.6.2 closed-loop diagnostic: EXACT governor actions first N steps, then V3.6 model.

Frozen V3.5 steering, frozen V3.6 longitudinal, no retraining, no IDM control.
All hybrid profiles are hand-coded interventions used for causal diagnosis ONLY.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parent.parent
DEP = ROOT / 'agent4_longitudinal_v36' / 'eval_longitudinal_v36.py'
if not DEP.is_file():
    raise RuntimeError(f'Missing sibling V3.6 dependency: {DEP}')
sys.path.insert(0, str(DEP.parent))
import eval_longitudinal_v36 as v36

STATUS = 'V362_EXACT_GOVERNOR_HANDOFF_DIAGNOSTIC'
PROFILES = ('governor_v35', 'learned_long_v36', 'handoff_80', 'handoff_160')
HANDOFF_STEPS = {'governor_v35': None, 'learned_long_v36': 0, 'handoff_80': 80, 'handoff_160': 160}


@contextmanager
def recorded_longitudinal(profile, long_model, pdd_features, actions):
    """Patch only longitudinal action for one episode; restore even on failure.

    An exact governor handoff executes the existing governor's returned action
    verbatim for ticks [0, N); model drives ticks [N, end), with no fallback.
    The frozen learned model is also queried in shadow during handoff to log
    what it *would* do, but these predictions do not affect the vehicle.
    """
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    v30 = v36.v30
    real_features = v30.feature_for_model
    real_geometry = v30.planned_geometry
    real_long = v30.governed_speed_action
    cached = {}
    handoff_steps = HANDOFF_STEPS[profile]

    def observe_features(env, max_gap):
        feat, route = real_features(env, max_gap)
        cached['features'] = feat
        cached['route'] = route
        return feat, route

    def observe_geometry(car, max_gap):
        geom = real_geometry(car, max_gap)
        cached['geom'] = geom
        return geom

    def apply_action(speed, base_speed, remaining):
        governor_action, target = real_long(speed, base_speed, remaining)
        idx = len(actions)
        if profile == 'governor_v35':
            predicted = None
            use_reference = True
        else:
            row = v36.make_row(
                cached['features'], pdd_features, cached['geom'],
                cached['route']['adapted'], target, cached['route']['remaining_m'])
            predicted = float(v36.predict_action(long_model, row))
            use_reference = idx < handoff_steps
        actual = float(governor_action if use_reference else predicted)
        actions.append({
            'step': idx,
            'governor_reference_action': float(governor_action),
            'learned_prediction': predicted,
            'actually_applied_action': actual,
            'reference_applied': use_reference,
            'speed_mps_before': float(speed),
            'target_speed_mps': float(target),
            'remaining_route_m': float(remaining),
        })
        return actual, float(target)

    v30.feature_for_model = observe_features
    v30.planned_geometry = observe_geometry
    v30.governed_speed_action = apply_action
    try:
        yield
    finally:
        v30.feature_for_model = real_features
        v30.planned_geometry = real_geometry
        v30.governed_speed_action = real_long


def run_one(env, map_name, seed, speed, profile, pdd, steering, long_model, steps, stride):
    actions = []
    with recorded_longitudinal(profile, long_model, pdd['features'], actions):
        item = v36.eval_one(env, map_name, seed, speed, 'student_v33', pdd, steering, steps, stride)
    n = item['summary']['n_applied_steps']
    item['map'] = map_name
    item['seed'] = seed
    item['speed_mps'] = speed
    item['profile'] = profile
    summary = item['summary']
    if len(actions) != n:
        item['errors'].append(f'LONGITUDINAL ALIGNMENT ERROR: {len(actions)} actions vs {n} steps')
    reference_steps = sum(a['reference_applied'] for a in actions[:n])
    learned_steps = n - reference_steps
    summary.update({
        'longitudinal_actions_aligned': len(actions) == n,
        'longitudinal_prediction_calls': sum(a['learned_prediction'] is not None for a in actions),
        'governor_reference_applied_steps': reference_steps,
        'learned_longitudinal_applied_steps': learned_steps,
        'hybrid_is_diagnostic_not_trained': profile.startswith('handoff_'),
        'handoff_step_requested': HANDOFF_STEPS[profile],
        'brake_applied_steps': sum(a['actually_applied_action'] < -0.02 for a in actions[:n]),
        'throttle_applied_steps': sum(a['actually_applied_action'] > 0.02 for a in actions[:n]),
        'mean_abs_speed_tracking_error_mps': (
            sum(abs(a['speed_mps_before'] - a['target_speed_mps']) for a in actions[:n])/n if n else None),
        'max_observed_speed_mps': max((a['speed_mps_before'] for a in actions[:n]), default=0),
        'max_observed_overspeed_mps': max((max(0, a['speed_mps_before']-a['target_speed_mps']) for a in actions[:n]), default=0),
        'speed_checkpoints_mps': {str(i): actions[i]['speed_mps_before'] for i in (0,10,20,30,40,60,80,100,120,160,200) if i < n},
    })
    for t in item.get('trace', []):
        i = t.get('step')
        if not isinstance(i,int) or i < 0 or i >= len(actions):
            continue
        a = actions[i]
        t['longitudinal_attribution_v362'] = {
            'profile': profile,
            'governor_reference_action': a['governor_reference_action'],
            'learned_prediction_shadow_or_applied': a['learned_prediction'],
            'reference_applied': a['reference_applied'],
            'actual_longitudinal_action': float(t['action_applied'][1]),
            'target_speed_mps': a['target_speed_mps'],
        }
        if abs(float(t['action_applied'][1])-a['actually_applied_action']) > 1e-8:
            item['errors'].append(f'Stepped action != recorded control at step {i}')
    summary['error_count'] = len(item['errors'])
    return item


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
    ap.add_argument('--out', type=Path, default=Path('agent4_v362_exact_handoff.json'))
    args = ap.parse_args(argv)
    try:
        maps = v36.parse_simple_csv(args.maps, v36.MAPS)
        seeds = v36.parse_seeds(args.seeds)
        speeds = v36.parse_speeds(args.speeds)
        profiles = v36.parse_simple_csv(args.profiles, PROFILES)
        if not 100 <= args.steps <= 5000 or not 1 <= args.trace_stride <= 100:
            raise ValueError('Invalid steps / trace-stride')
    except ValueError as exc:
        ap.error(str(exc))
    pdd = joblib.load(args.pdd)
    steering = joblib.load(args.steering)
    longitudinal = joblib.load(args.longitudinal)
    if pdd.get('label_mode') != 'at_t' or len(pdd.get('features', [])) != 15:
        raise ValueError('Expected PDD V2 at_t with 15 features')
    v36.check_student(steering, pdd['features'], seeds, 'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    if longitudinal.get('version') != v36.VERSION or longitudinal.get('feature_order') != v36.names(pdd['features']):
        raise ValueError('Wrong V3.6 version or feature order')
    if longitudinal.get('pdd_features') != pdd['features']:
        raise ValueError('PDD feature mismatch')
    v36.check_eval_seeds(longitudinal, seeds)
    if not set(speeds) <= set(longitudinal['speed_targets']):
        raise ValueError('Requested speed was not part of V3.6 training')

    from metadrive.envs.metadrive_env import MetaDriveEnv
    report = {
        'status': STATUS,
        'maps': maps, 'seeds': seeds, 'speeds_mps': speeds, 'profiles': profiles,
        'handoff_steps': {p: HANDOFF_STEPS[p] for p in profiles},
        'frozen_steering_v35': True, 'frozen_longitudinal_v36': True,
        'new_training': False, 'idm_used_for_control': False, 'traffic_density': 0.0,
        'warnings': [
            'This is a diagnostic intervention. Handoff profiles are NOT fully learned control.',
            'Governor action is used verbatim for exactly the first N steps, then V3.6 acts without fallback.',
            'During governor-controlled steps the learned longitudinal model is queried in shadow only.',
            'Seeds 76-77 have already been evaluated; DO NOT use these runs to claim held-out generalization.',
            'MetaDrive state is privileged; road margin is approximate; no traffic.',
        ],
        'episodes': [], 'success_by_profile': {}, 'scores_by_speed_map_profile': [],
    }
    for m in maps:
        env = None
        try:
            env = MetaDriveEnv(dict(use_render=False, map=m, num_scenarios=max(seeds)+1,
                                    traffic_density=0.0, horizon=max(1000, args.steps+10), log_level=40))
            for seed in seeds:
                for speed in speeds:
                    for p in profiles:
                        ep = run_one(env, m, seed, speed, p, pdd, steering, longitudinal,
                                     args.steps, args.trace_stride)
                        report['episodes'].append(ep)
                        s = ep['summary']
                        print(f'{m} seed={seed} {speed:g}m/s {p}: {s["status"]}; '
                              f'{s["travelled_approx_m"]:.1f}m; '
                              f'gov_steps={s["governor_reference_applied_steps"]}; '
                              f'learned_steps={s["learned_longitudinal_applied_steps"]}', flush=True)
        finally:
            if env is not None:
                env.close()
    report['success_by_profile'] = {
        p: {'success': sum(e['profile'] == p and e['summary']['status'] == 'arrive_dest' for e in report['episodes']),
            'total': sum(e['profile'] == p for e in report['episodes'])} for p in profiles
    }
    report['scores_by_speed_map_profile'] = v36.group_scores(v36.summary_rows(report['episodes']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('Results:', args.out.resolve())
    print('Success:', report['success_by_profile'])


if __name__ == '__main__':
    main()
