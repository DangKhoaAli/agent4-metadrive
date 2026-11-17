"""Validate inference plumbing without letting an uncalibrated model drive.

Run:
 python dryrun_agent4_v2.py ./pdd_v2_models/at_t/model_v2.joblib

The generated predictions are logged but NOT fed to env.step().
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from shared_adapter_v2 import FEATURES, metadrive_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--out', type=Path, default=Path('agent4_dryrun_v2.json'))
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    from metadrive import MetaDriveEnv

    bundle = joblib.load(args.model)
    if bundle.get('features') != FEATURES:
        raise ValueError('Model and Adapter have different feature order')
    env = MetaDriveEnv(dict(use_render=False, traffic_density=0.2, num_scenarios=1, log_level=40))
    rows = []
    try:
        env.reset(seed=0)
        for t in range(args.steps):
            features, _ = metadrive_features(env)
            x = np.asarray([[features[k] for k in FEATURES]], dtype=np.float64)
            a1 = float(bundle['models']['steering'].predict(x)[0])
            a2 = float(bundle['models']['longitudinal'].predict(x)[0])
            rows.append({'step': t, 'features': features,
                         'predicted_steer_carla_candidate': a1,
                         'predicted_longitudinal_metadrive_candidate': a2})
            # Crucial: do NOT apply model predictions without calibration.
            _, _, terminated, truncated, _ = env.step([0.0, 0.30])
            if terminated or truncated:
                break
    finally:
        env.close()
    report = {'status': 'DRY_RUN_NO_MODEL_ACTION_APPLIED', 'samples': rows,
              'warning': 'Predictions are UNCALIBRATED; step commands use [0,0.30], not model outputs.'}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Dry-run predictions:', len(rows), 'Saved:', args.out)


if __name__ == '__main__':
    main()
