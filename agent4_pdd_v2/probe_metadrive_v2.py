"""Smoke-test MetaDrive 0.4.3 physical-feature adapter, NO trained policy driving.

Examples:
  python probe_metadrive_v2.py --pdd-v2 ./pdd_v2_output
  python probe_metadrive_v2.py --seeds 0 1 2 --steps 50 --traffic-density 0.25

Produces probe_v2.json containing feature samples and distribution comparison.
MetaDrive env is never controlled by the offline model in this script.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from shared_adapter_v2 import FEATURES, metadrive_features


def stats(xs):
    a = sorted(xs)
    if not a:
        return None
    return {'min': a[0], 'p10': a[int(.1*(len(a)-1))], 'median': a[int(.5*(len(a)-1))],
            'p90': a[int(.9*(len(a)-1))], 'max': a[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--traffic-density', type=float, default=0.20)
    ap.add_argument('--pdd-v2', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=Path('metadrive_v2_probe.json'))
    args = ap.parse_args()
    if args.steps < 1 or not (0 <= args.traffic_density <= 1):
        ap.error('--steps must be positive and --traffic-density in [0,1]')
    from metadrive import MetaDriveEnv

    values = defaultdict(list)
    first_samples, errors, episodes = [], [], []
    cfg = dict(use_render=False, traffic_density=args.traffic_density, num_scenarios=max(1, len(args.seeds)),
               log_level=40)
    env = None
    try:
        env = MetaDriveEnv(cfg)
        for seed in args.seeds:
            obs, info = env.reset(seed=seed)
            last_info = {}
            one = {'seed': seed, 'steps': 0}
            for t in range(args.steps):
                try:
                    feats, details = metadrive_features(env)
                    if list(feats) != FEATURES:
                        raise AssertionError('Adapter feature order mismatch')
                    if not all(math.isfinite(float(v)) for v in feats.values()):
                        raise AssertionError('NaN/Inf in MetaDrive feature output')
                    for k, v in feats.items():
                        values[k].append(float(v))
                    if len(first_samples) < 8:
                        first_samples.append({'seed': seed, 'step': t, 'features': feats, 'diagnostics': details})
                except Exception as ex:
                    errors.append(f'seed={seed}, step={t}: {type(ex).__name__}: {ex}')
                    break
                # Gentle throttle / straight only to sample states. NOT a driving policy.
                _, _, terminated, truncated, last_info = env.step([0.0, 0.3])
                one['steps'] += 1
                if terminated or truncated:
                    break
            one['last_step_info'] = {k: last_info.get(k) for k in ('arrive_dest','out_of_road','crash','route_completion')}
            episodes.append(one)
            if errors:
                break
    finally:
        if env is not None:
            env.close()

    report = {
        'status': 'METADRIVE_V2_ADAPTER_PROBE_NOT_CLOSED_LOOP_VALIDATION',
        'python_simulator_version': 'read from installed MetaDrive during user run',
        'configuration': cfg,
        'feature_order': FEATURES,
        'n_samples': len(values.get('speed_mps', [])),
        'feature_stats_metadrive': {k: stats(values[k]) for k in FEATURES},
        'first_samples': first_samples,
        'episodes': episodes,
        'errors': errors,
        'warnings': [],
        'limitations': [
            'Agent 4 model was NOT used in this probe.',
            'Object data are privileged simulator state; not inferred from RGB camera.',
            'Navigation checkpoints are clipped to 50m, may differ from CARLA route planner.',
            'Nearby traffic may be absent in some sampled seeds/steps.',
        ],
    }
    if args.pdd_v2 is not None:
        meta_file = args.pdd_v2 / 'metadata_v2.json'
        m = json.loads(meta_file.read_text(encoding='utf-8-sig'))
        if m.get('feature_order') != FEATURES:
            raise ValueError('PDD V2 metadata schema differs from MetaDrive adapter')
        report['feature_stats_pdd'] = m['feature_stats']
        discrepancies = {}
        for feature in FEATURES:
            a = report['feature_stats_metadrive'][feature]
            b = m['feature_stats'].get(feature)
            if a and b and a['median'] is not None and b['median'] is not None:
                discrepancies[feature] = {'pdd_median': b['median'], 'metadrive_median': a['median'],
                                          'absolute_median_gap': abs(a['median']-b['median'])}
        report['distribution_median_gaps'] = discrepancies
        # Numerical sanity flags are diagnostics, not pass/fail proof.
        for key, threshold in [('speed_mps', 8.0), ('wp1_forward_m', 20.0),
                               ('wp2_forward_m', 20.0), ('lane_width_m', 1.0)]:
            d = discrepancies.get(key)
            if d is not None and d['absolute_median_gap'] > threshold:
                report['warnings'].append(
                    f'{key}: PDD vs MetaDrive median gap {d["absolute_median_gap"]:.2f} > {threshold:.2f}; revisit feature equivalence'
                )
    if values.get('has_front_vehicle') and sum(values['has_front_vehicle']) == 0:
        report['warnings'].append('No front vehicles sampled: try more seeds/steps or higher traffic density.')
    dest = args.out.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print('MetaDrive V2 samples:', report['n_samples'])
    print('Errors:', errors[:2])
    print('Warnings:', report['warnings'][:5])
    print('Saved:', dest)
    if errors:
        raise RuntimeError('MetaDrive V2 adapter failed; see JSON for details')
    if report['n_samples'] == 0:
        raise RuntimeError('No MetaDrive feature samples produced')
    print('PASS: shape, feature order and finite values. NOT a PDD-to-MetaDrive policy proof.')


if __name__ == '__main__':
    main()
