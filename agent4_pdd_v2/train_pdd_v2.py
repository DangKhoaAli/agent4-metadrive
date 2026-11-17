"""Train independent steering and longitudinal regression pilots on PDD V2.

Run: python train_pdd_v2.py ./pdd_v2_output --out ./pdd_v2_models
Use --label at_t (instantaneous candidate) or --label mean5 (window average).

Validation selects each head; test evaluated only once after selection.
This DOES NOT certify cross-simulator control / obstacle avoidance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared_adapter_v2 import FEATURES

TARGETS = {
    'at_t': ('steer_at_t_carla', 'a2_at_t_metadrive_candidate'),
    'mean5': ('steer_mean_next5_carla', 'a2_mean_next5_metadrive_candidate'),
}


def read_split(folder: Path, split: str, labels):
    df = pd.read_csv(folder / (split + '.csv'), encoding='utf-8-sig')
    need = FEATURES + list(labels) + ['scenario', 'driver']
    missing = [x for x in need if x not in df]
    if missing:
        raise RuntimeError(f'{split}.csv missing fields: {missing}')
    numeric = df[FEATURES + list(labels)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f'{split}.csv contains nonnumeric, Inf or NaN')
    return df, numeric[:, :len(FEATURES)], numeric[:, len(FEATURES):]


def score(y_true, y_pred, is_steering: bool, scale: float, report_details=True):
    err = np.abs(y_true - np.clip(y_pred, -1, 1))
    hard = np.abs(y_true) >= 0.05 if is_steering else y_true < -0.05
    result = {'mae': float(err.mean()), 'rmse': float(np.sqrt(np.mean((y_true-y_pred)**2))),
              'hard_samples': int(hard.sum()),
              'hard_mae': float(err[hard].mean()) if hard.any() else None}
    # Weighted validation score penalizes hard cases when they occur; based on TRAIN scale.
    normalized = result['mae'] / max(scale, 1e-6)
    hard_norm = result['hard_mae'] / max(scale, 1e-6) if hard.any() else normalized
    result['selection_score'] = float(0.7 * normalized + 0.3 * hard_norm)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data_dir', type=Path)
    ap.add_argument('--out', type=Path, default=Path('pdd_v2_models'))
    ap.add_argument('--label', choices=['at_t', 'mean5'], default='at_t')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    folder = args.data_dir.resolve()
    metadata = json.loads((folder / 'metadata_v2.json').read_text(encoding='utf-8-sig'))
    if metadata['feature_order'] != FEATURES:
        raise RuntimeError('Adapter feature contract differs from train CSV metadata')
    if metadata['status'] != 'V2_PROVISIONAL_SHARED_FEATURES_NOT_CLOSED_LOOP_VALIDATED':
        raise RuntimeError('Unexpected metadata status')
    labels = TARGETS[args.label]
    tr_df, x_tr, y_tr = read_split(folder, 'train', labels)
    va_df, x_va, y_va = read_split(folder, 'val', labels)
    te_df, x_te, y_te = read_split(folder, 'test', labels)
    groups = [set(df['scenario']) for df in (tr_df, va_df, te_df)]
    if any(groups[i].intersection(groups[j]) for i in range(3) for j in range(i+1,3)):
        raise RuntimeError('Scenario leakage across splits')

    chosen = {}
    details = {}
    baseline = {}
    y_scale = y_tr.std(axis=0)
    for dim, name in enumerate(('steering', 'longitudinal')):
        hard = np.abs(y_tr[:, dim]) >= .05 if dim == 0 else y_tr[:, dim] < -0.05
        weights = np.ones(len(x_tr), dtype=float)
        weights[hard] = 3.0 if dim == 0 else 8.0  # counter rare turns/braking; tune on validation only
        models = {
            'ridge': Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=10.0))]),
            'hist_gradient_boost': Pipeline([('scaler', StandardScaler()), ('regressor', HistGradientBoostingRegressor(
                max_iter=160, learning_rate=.05, max_leaf_nodes=15,
                min_samples_leaf=25, l2_regularization=.5, early_stopping=False,
                random_state=args.seed))]),
        }
        vals = {}
        for model_name, model in models.items():
            model.fit(x_tr, y_tr[:, dim], regressor__sample_weight=weights)
            prediction = model.predict(x_va)
            vals[model_name] = score(y_va[:, dim], prediction, dim == 0, y_scale[dim])
        # Add train-only median, robust to outlier target distribution.
        const = np.median(y_tr[:, dim])
        base_va = score(y_va[:, dim], np.repeat(const, len(y_va)), dim == 0, y_scale[dim])
        winner = min(vals, key=lambda model: vals[model]['selection_score'])
        chosen[name] = models[winner]
        pred_test = chosen[name].predict(x_te)
        details[name] = {
            'chosen': winner,
            'validation': {'constant_median': base_va, **vals},
            'test': {'constant_median': score(y_te[:, dim], np.repeat(const, len(y_te)), dim==0, y_scale[dim]),
                     'selected': score(y_te[:, dim], pred_test, dim==0, y_scale[dim])},
            'train_hard_samples': int(hard.sum()),
        }
    out = args.out.resolve() / args.label
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'models': chosen, 'features': FEATURES, 'label_mode': args.label,
                 'output_labels': labels, 'WARNING': 'PILOT MODEL ONLY; calibration not performed'},
                out / 'model_v2.joblib')
    report = {'status': 'V2_OFFLINE_ONLY_NOT_METADRIVE_CALIBRATED', 'labels': labels, 'features': FEATURES,
              'n_rows': {'train':len(tr_df),'val':len(va_df),'test':len(te_df)},
              'results': details, 'training': 'Two independent regression heads with hard-sample weighting',
              'limits': ['No closed-loop driving tested', 'No calibrated CARLA->MetaDrive action',
                         'Object/waypoint representations are candidate semantic matches',
                         'Extra physical features may not generalize to different traffic density']}
    (out / 'metrics_v2.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Finished V2 offline training:')
    for name, info in details.items():
        v = info['validation']['constant_median']['selection_score']
        vv = info['validation'][info['chosen']]['selection_score']
        print(f"  {name}: chosen={info['chosen']}, val selection score baseline={v:.4f} candidate={vv:.4f}")
    print('Model:', out / 'model_v2.joblib')
    print('Metrics:', out / 'metrics_v2.json')
    print('CAUTION: model is not ready to drive until MetaDrive V2 probe & action calibration pass.')


if __name__ == '__main__':
    main()
