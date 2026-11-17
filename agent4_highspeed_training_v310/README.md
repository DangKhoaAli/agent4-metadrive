# Agent 4 V3.10 — High-Speed Learned Steering (MetaDrive)

## Objective and truthfulness

Train a **new model** to predict **steering residual** at 4, 6, 8, and 10 m/s, from on-teacher-policy MetaDrive rollouts, using frozen PDD V2 as a base. Does not reuse the failing V3.5 steering model.

This is **Behavior Cloning / teacher distillation**, *not* RL. The teacher is `rule_lateral` (`V3.1` lateral and heading feedback), used **only during data collection**. The model is a zero-intercept, smooth `Ridge` regressor trained on states including perturbations: 6 basis features comprising lateral offset, heading error and speed interactions. Model predictions directly control steering in closed-loop evaluation; no rule-teacher steering fallback. The usual rule-based speed governor remains for longitudinal control.

**PDD V2 is a learned steering base, not the V3.5 student.** The 15 frozen PDD features and V3.3's 4 extra state feature names define the input contract. The V3.10 estimator intentionally uses lateral/heading/speed for the correction to prevent accumulation of an arbitrary constant steering bias. No PDD/teacher retraining.

## What the prior V3.9.2 data support

`SSSSS`, seeds 124–125, 8 & 10 m/s: `pdd_only` 4/4, `rule_lateral` 4/4 successful. V3.10 must match **both** and should not be deployed just because it mimics teacher in MAE. Teacher max lateral offset below ~2.4 cm, while baseline PDD reaches ~28.4 cm at 10 m/s, still succeeds. These seeds are **not** used for V3.10 training.

## Directory structure

Extract `agent4_highspeed_training_v310/` next to:

- `agent4_controller_v30/`, `agent4_controller_v31/`, `agent4_calibration_v27/`, `agent4_diagnostic_v29/` and their dependencies
- `agent4_learned_residual_v33/`
- `agent4_robustness_v34/` / `agent4_steering_diagnostic_v391/` (optional for V3.10)
- `pdd_v2_models/at_t/model_v2.joblib`

Run PowerShell in the parent `dataset` directory. Requires **MetaDrive 0.4.3**, Python 3.10, joblib, numpy, sklearn. Offline tests do not require MetaDrive runtime.

### 0 — Offline tests

```powershell
python agent4_highspeed_training_v310\test_v310.py
```

### 1 — Smoke teacher data collection (separate folder and seeds)

```powershell
python agent4_highspeed_training_v310\collect_v310.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --maps S --speeds 4 --train-seeds 130 --val-seeds 142 --eval-seeds 146 --steps 1000 --out-dir ".\agent4_v310_smoke_data"
```

Inspect `agent4_v310_smoke_data/manifest.json`: at least some `arrive_dest`, saved rows, `noise_steps>0`, `recovery_state_steps` and errors. **Smoke folder is separate and is not the intended training dataset.**

### 2 — Full data collection

```powershell
python agent4_highspeed_training_v310\collect_v310.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --maps S,SSSSS --speeds 4,6,8,10 --train-seeds 130-141 --val-seeds 142-145 --eval-seeds 146-149 --steps 2500 --perturb-md 0.0035 --pulse-steps 6 --out-dir ".\agent4_v310_steering_data"
```

Counts: `2 maps × 4 speeds × (12 train + 4 val) = 128 teacher episodes`. Evaluation seeds 146–149 are reserved and **never collected**. Rows from unsuccessful episodes discarded, failures recorded. A truly large lateral recovery state is **not guaranteed** by these small perturbations; inspect `recovery_state_steps` and the training metrics before extrapolating. If several cells contain no successes, **do not train**; diagnose instead. If an output folder already contains files, use a fresh folder name to avoid silent overwrite.

### 3 — Fit new learned residual steering

```powershell
python agent4_highspeed_training_v310\train_v310.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --data-dir ".\agent4_v310_steering_data" --out-model ".\agent4_v310_steering.joblib" --out-metrics ".\agent4_v310_training_metrics.json"
```

The validation report compares **PDD vs learned** teacher-action error on **held-out validation seeds**. A low MAE is not sufficient proof; do not accept a model that regresses away steering correction when disturbed. `basis_coefficients` and `recovery_coverage` help diagnose this.

### 4 — Independent closed-loop evaluation

```powershell
python agent4_highspeed_training_v310\eval_v310.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v310_steering.joblib" --maps S,SSSSS --seeds 146,147 --speeds 4,6,8,10 --profiles pdd_only,rule_lateral,learned_v310 --steps 2500 --out ".\agent4_v310_closed_loop_eval.json"
```

Runs `2 maps × 2 seeds × 4 speeds × 3 profiles = 48 episodes`. An episode is successful **only if `arrive_dest`**, and model attribution includes `student_predictions_aligned` and direct applied actions. No rule steering fallback, no IDM, no learned longitudinal. Results also written to `.csv`.

### 5 — How to choose next model

- Prefer the learned model only if closed-loop arrival rate and lateral margin are **at least as good** as PDD on *independent* seeds.
- Compare at matched speed and route, report time-to-goal and actual terminal status; avoid selecting only by offline MAE.
- Then expand validation to curved C and X at **lower safe speeds first**, and introduce faster lane changes and on-policy DAgger-style corrections only with defensible labels and outcome checks.
- A vehicle capable of 10 m/s on `SSSSS` is **not** certified for 10 m/s on C/X or with traffic.

Research simulation only. **Not suitable for deployment on physical vehicles.**
