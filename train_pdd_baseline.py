"""Train a *pilot*, offline PDD steering/longitudinal-control baseline.

Reads the exact CSVs produced by prepare_pdd_pilot.py:
  pdd_pilot_output/{train,val,test}.csv and metadata.json

Outputs to --out/<label>/:
  model_bundle.joblib     (feature scaler, trained regression model, target scaler)
  metrics.json            (validation and held-out test metrics)

LIMITATIONS:
  - Alignment and CARLA->MetaDrive transfer are NOT certified.
  - This is NOT a MetaDrive-ready agent. Do not feed raw 259D/19D obs into it.
  - No test data participates in model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {
    "at_t": ["steer_at_t_carla", "a2_at_t_metadrive_candidate"],
    "mean5": ["steer_mean_next5_carla", "a2_mean_next5_metadrive_candidate"],
}


def load_split(folder: Path, name: str, features: list[str], targets: list[str]):
    path = folder / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run prepare_pdd_pilot.py first.")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    needed = features + targets + ["scenario", "driver"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    numeric = df[features + targets].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    if not valid.all():
        raise ValueError(f"{path.name}: {int((~valid).sum())} rows have nonnumeric/NaN/Inf features/labels")
    return df, numeric[features].to_numpy(dtype=np.float64), numeric[targets].to_numpy(dtype=np.float64)


def assess(y_true: np.ndarray, y_pred: np.ndarray, y_scale: np.ndarray):
    y_pred = np.clip(y_pred, -1.0, 1.0)
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true), axis=0))
    strong_turn = np.abs(y_true[:, 0]) >= 0.05
    braking = y_true[:, 1] < -0.05
    result = {
        "n": int(len(y_true)),
        "mae_steer": round(float(mae[0]), 6),
        "mae_longitudinal": round(float(mae[1]), 6),
        "rmse_steer": round(float(rmse[0]), 6),
        "rmse_longitudinal": round(float(rmse[1]), 6),
        "mean_normalized_mae": round(float(np.mean(mae / np.maximum(y_scale, 1e-6))), 6),
        "strong_turn_samples": int(strong_turn.sum()),
        "braking_samples": int(braking.sum()),
        "strong_turn_mae_steer": (round(float(np.mean(np.abs(y_pred[strong_turn, 0] - y_true[strong_turn, 0]))), 6)
                                  if strong_turn.any() else None),
        "braking_mae_longitudinal": (round(float(np.mean(np.abs(y_pred[braking, 1] - y_true[braking, 1]))), 6)
                                      if braking.any() else None),
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", type=Path, help="Folder with train.csv, val.csv, test.csv, metadata.json")
    p.add_argument("--out", type=Path, default=Path("pdd_baseline_output"))
    p.add_argument("--label", choices=TARGETS.keys(), default="at_t",
                   help="at_t: action at telemetry step; mean5: 0.25-second control mean")
    p.add_argument("--max-iter", type=int, default=250, help="MLP optimizer max iterations")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    folder = args.data_dir.expanduser().resolve()
    meta_path = folder / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    if meta.get("status") != "PILOT_ONLY_LABEL_ALIGNMENT_AND_TRANSFER_NOT_PROVEN":
        raise ValueError("Unexpected dataset metadata; inspect it before training")
    features = meta["input_features"]
    targets = TARGETS[args.label]
    if not set(targets).issubset(meta.get("candidate_training_labels", [])):
        raise ValueError("Chosen labels not declared in metadata")

    df_tr, x_tr, y_tr = load_split(folder, "train", features, targets)
    df_va, x_va, y_va = load_split(folder, "val", features, targets)
    df_te, x_te, y_te = load_split(folder, "test", features, targets)
    groups = [set(df.scenario.unique()) for df in (df_tr, df_va, df_te)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Scenario leakage detected across train/val/test")
    if len(df_tr) == 0 or len(df_va) == 0 or len(df_te) == 0:
        raise ValueError("Empty train/val/test split")

    y_scaler = StandardScaler().fit(y_tr)
    y_tr_scaled = y_scaler.transform(y_tr)
    y_std = np.maximum(y_scaler.scale_, 1e-6)

    # Both candidate models train only on train.csv.
    models = {
        "ridge": Pipeline([
            ("x_scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]),
        "mlp": Pipeline([
            ("x_scaler", StandardScaler()),
            ("regressor", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=args.max_iter,
                early_stopping=False,  # Avoid random-frame internal validation leakage
                random_state=args.seed,
                alpha=0.001,
                learning_rate_init=0.001,
                n_iter_no_change=30,
            )),
        ]),
    }
    val_scores = {}
    for name, model in models.items():
        model.fit(x_tr, y_tr_scaled)
        pred = y_scaler.inverse_transform(model.predict(x_va))
        val_scores[name] = assess(y_va, pred, y_std)

    # Train-only mean baseline; not a trained driving policy.
    const_val_pred = np.repeat(y_tr.mean(axis=0, keepdims=True), len(y_va), axis=0)
    val_baseline = assess(y_va, const_val_pred, y_std)
    winner = min(models, key=lambda n: val_scores[n]["mean_normalized_mae"])
    best = models[winner]

    # Crucially, test.csv is evaluated only after selection using val.csv.
    test_pred = y_scaler.inverse_transform(best.predict(x_te))
    test_scores = assess(y_te, test_pred, y_std)
    const_test_pred = np.repeat(y_tr.mean(axis=0, keepdims=True), len(y_te), axis=0)
    test_baseline = assess(y_te, const_test_pred, y_std)

    out = args.out.expanduser().resolve() / args.label
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / "model_bundle.joblib"
    joblib.dump({
        "model": best,
        "y_scaler": y_scaler,
        "features": features,
        "targets": targets,
        "label_mode": args.label,
        "source_metadata_status": meta["status"],
        "warning": "OFFLINE PILOT ONLY; not calibrated to MetaDrive; no Agent 4 adapter included",
    }, bundle_path)

    report = {
        "status": "OFFLINE_PDD_BASELINE_NOT_METADRIVE_VALIDATED",
        "label_mode": args.label,
        "input_features_in_order": features,
        "output_labels_in_order": targets,
        "rows": {"train": len(df_tr), "val": len(df_va), "test": len(df_te)},
        "split_scenarios": {"train": sorted(groups[0]), "val": sorted(groups[1]), "test": sorted(groups[2])},
        "train_only_target_mean": y_tr.mean(axis=0).tolist(),
        "train_only_target_std": y_std.tolist(),
        "validation": {"constant_mean": val_baseline, **val_scores},
        "selected_using_validation": winner,
        "test": {"constant_mean": test_baseline, "selected_model": test_scores},
        "model_path": str(bundle_path),
        "limitations": [
            "Only PDD offline imitation quality was tested; closed-loop route completion is NOT measured.",
            "PDD image/control timing, CARLA vs MetaDrive action calibration, waypoint coordinate equivalence remain unverified.",
            "MetaDrive physical observation adapter (including lead vehicle) has NOT been implemented.",
            "Model can lack obstacle/side-vehicle/traffic-light signals; good MAE alone is insufficient.",
            "The at_t and mean5 targets have different meanings; DO NOT directly rank their MAEs against each other.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Rows: train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}")
    print(f"Validation normalized MAE: mean={val_baseline['mean_normalized_mae']}, "
          f"ridge={val_scores['ridge']['mean_normalized_mae']}, mlp={val_scores['mlp']['mean_normalized_mae']}")
    print(f"Chosen using validation: {winner}")
    print(f"Test normalized MAE: mean={test_baseline['mean_normalized_mae']}, "
          f"chosen={test_scores['mean_normalized_mae']}")
    print(f"Test strong turns: {test_scores['strong_turn_samples']}; braking samples: {test_scores['braking_samples']}")
    print(f"Saved: {bundle_path}")
    print(f"Saved: {out / 'metrics.json'}")
    print("WARNING: This model is NOT ready to drive in MetaDrive without an observation adapter and calibration.")


if __name__ == "__main__":
    main()
