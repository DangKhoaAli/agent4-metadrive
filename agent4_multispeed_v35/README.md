# Agent 4 V3.5 — Multispeed learned steering residual

## Why V3.5?

The uploaded V3.4 speed stress test (C/X, seeds 36–39, no traffic) found:

| Speed | V3.1 hand-written correction | V3.3 learned steering residual |
|---|---:|---:|
| 2 m/s | 1/8 | 5/8 |
| 4 m/s | 2/8 | 0/8 |

V3.3 had been trained only at **3 m/s**. This V3.5 trains a **new steering residual model** on IDM steering labels at several speeds. It does not yet learn throttle/brake. PDD V2 stays frozen.

### Important model-input detail

Existing PDD input `speed_mps` (measured actual speed) is already a model feature. We **preserve the exact V3.3 19-feature order** (15 PDD + 4 extras) to keep the closed-loop V3.4 adapter compatible. `target_speed_mps` is added only to teacher CSV as **audit/training-stratification metadata**, **NOT** a model input. This means the learner sees current speed but not the governor's desired speed.

Teacher IDM computes steering while an independent governor controls speed, same as V3.3. Training balances `(target speed, map)` groups so longer episodes don't automatically dominate. Only teacher episodes reaching `arrive_dest` are retained. A nonzero speed perturbation affects the executed car action only, never the IDM label. Reduced `--perturb-md` may improve successful expert data collection, but inspect failure causes.

## Project directory

Extract `agent4_multispeed_v35` as a sibling of:

- `agent4_learned_residual_v33`
- `agent4_robustness_v34`
- `agent4_controller_v30`, `agent4_controller_v31`
- Previous V3.3 dependencies (V2.6 etc.)
- `pdd_v2_models/at_t/model_v2.joblib`, `agent4_v33_student.joblib`

Install Python 3.10, MetaDrive 0.4.3, numpy, joblib, scikit-learn. Run commands in the `dataset` directory.

## 0. Offline synthetic tests

```powershell
python agent4_multispeed_v35\test_v35.py
```

Expected: `... PASS`. Does not require MetaDrive or existing model files.

## 1. Collect **new** IDM teacher data at 2, 3, 4 m/s

These seeds are **new** and segregated:
- Train: 40–51
- Validation: 52–55
- Held-out evaluation: 56–59 (NEVER collected)
- Previous benchmark seeds 0–39 remain excluded.

```powershell
python agent4_multispeed_v35\collect_multispeed_v35.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps S,C,X --speeds 2,3,4 --train-seeds 40-51 --val-seeds 52-55 --eval-seeds 56-59 --steps 1800 --perturb-md 0.008 --out-dir ".\agent4_v35_multispeed_data"
```

Collection runs the existing V3.3 aligned IDM teacher sampler once per requested speed. Output:

```
agent4_v35_multispeed_data/
  speed_2p000/ train.csv val.csv manifest.json
  speed_3p000/ train.csv val.csv manifest.json
  speed_4p000/ train.csv val.csv manifest.json
  train.csv
  val.csv
  manifest.json
```

**Inspect** `manifest.json` -> `success_by_split_speed_map` before training. The training script rejects a speed/map cell with fewer than 100 training or 50 validation rows. IDM teacher failures do **not** count as model failures, and their rows are not included. All speed/map groups must have coverage, especially 4 m/s.

If collection fails mid-run, fix root cause, then re-run with `--reuse-existing`: it will reuse complete per-speed datasets and strictly validate their seed/speed/map/perturbation contracts. If a specific speed directory was only partially created, delete **that incomplete directory only** before restarting.

## 2. Fit new steering model

```powershell
python agent4_multispeed_v35\train_multispeed_v35.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --data-dir ".\agent4_v35_multispeed_data" --out-model ".\agent4_v35_student.joblib" --out-metrics ".\agent4_v35_training_metrics.json"
```

The model is a HistGradientBoostingRegressor learning teacher minus PDD steering, with equal aggregate sample weights per speed/map, using exactly the same 19 observation features as V3.3. Evaluate `val` metrics **separately by target speed and map**. Offline MAE alone cannot establish that the student drives correctly.

## 3. Closed-loop holdout A/B on entirely new seeds

First pilot two held-out seeds, maps C/X, all three speeds, two learned models:

```powershell
python agent4_multispeed_v35\evaluate_multispeed_v35.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --old-student ".\agent4_v33_student.joblib" --student ".\agent4_v35_student.joblib" --maps C,X --seeds 56,57 --speeds 2,3,4 --profiles student_v33,student_v35 --steps 1800 --out ".\agent4_v35_closed_loop_eval.json"
```

That is **24 actual MetaDrive episodes**. Compare V3.3 against V3.5 on the same seed and speed. If justified, expand to held-out seeds 58,59. Do not move held-out seeds into training/tuning after seeing results; reserve further unseen seeds for subsequent evaluation.

Student steering remains **PDD V2 + learned residual**; throttle/brake still come from the **rule-based speed governor**. No IDM is used for steering or longitudinal control during closed-loop student evaluation. `lane_departure_guard` is a precautionary stop, *not* an authoritative `out_of_road` report. MetaDrive's line contact flags are authoritative for `out_of_road`.

## 4. Files to send back

- `agent4_v35_multispeed_data/manifest.json`
- `agent4_v35_training_metrics.json`
- `agent4_v35_closed_loop_eval.json`

## Scope / limitations

- Never use on a physical vehicle. Privileged MetaDrive state, no traffic, simulator-only.
- IDM success only means the data collection teacher succeeds; it is not learned-model success.
- V3.5 aims to correct steering speed sensitivity, **not** to learn longitudinal control.
- A speed-independent policy that fails at 4 m/s cannot be made safe by assuming a good offline MAE.
- Speed 4 m/s performance remains **unverified** until the new model is evaluated closed-loop.
