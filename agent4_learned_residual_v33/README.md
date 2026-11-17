# Agent 4 V3.3 — Frozen PDD + learned steering residual (MetaDrive-only research)

## Objective
V3.2 demonstrated that PDD steering plus hand-tuned lateral/heading correction
is brittle on C seed2 and X seed2. Instead of another gain sweep, V3.3 **trains
an actual residual model** on expert **MetaDrive IDM steering** at the **same
3 m/s** as the closed-loop experiments. The frozen CARLA PDD V2 model remains
the base and supplies one student input (`base_pdd_steer_md`). This is **new
MetaDrive-supervised data**, not CARLA PDD ground truth.

Teacher: a subclass of MetaDrive `IDMPolicy` computes steering; a separate speed
governor replaces longitudinal control. Optional small deterministic steering
pulses perturb only the **executed** steering to expose off-centre recovery
states; labels remain the original unperturbed IDM steering (clipped ±0.35).
Only episodes reaching `arrive_dest` are retained. Strict pre-step action
alignment is checked. All per-step observations are privileged simulator data.

Student: 15 PDD features + `base_pdd_steer_md`, `planned_lateral_right_m`,
`heading_error_rad`, `adaptive_lookahead`. HistGradientBoostingRegressor learns
bounded steering residual from teacher action minus frozen PDD prediction.
At runtime the learned model provides the residual; **no rule-based
lateral/heading correction is applied**. A separate fixed speed governor is
still used, so Agent 4 is NOT fully end-to-end autonomous driving.

## Dependencies / layout
Extract next to these existing folders, **not inside** them:

```
dataset/
  agent4_pdd_v2/
  agent4_calibration_v22/
  agent4_calibration_v23/
  agent4_calibration_v24/
  agent4_calibration_v26/
  agent4_calibration_v27/
  agent4_diagnostic_v29/
  agent4_controller_v30/
  agent4_controller_v31/
  agent4_learned_residual_v33/
  pdd_v2_models/at_t/model_v2.joblib
```

Python 3.10 with MetaDrive 0.4.3, numpy, joblib, scikit-learn.

## Run (PowerShell in dataset/)

0. Offline tests (do not need MetaDrive):

```powershell
python agent4_learned_residual_v33\test_v33.py
```

1. Collect **new MetaDrive IDM teacher data** from held-out maps/seeds. Train
4–23, validation 24–27; eval 28–31 is reserved. Leave the reserved eval seeds
*outside* the collector:

```powershell
python agent4_learned_residual_v33\collect_teacher_v33.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps S,C,X --train-seeds 4-23 --val-seeds 24-27 --eval-seeds 28-31 --steps 1400 --speed-mps 3 --perturb-md 0.012 --out-dir ".\agent4_v33_teacher_data"
```

Output: `agent4_v33_teacher_data/train.csv`, `val.csv`, `manifest.json`.
`manifest.json` records success counts and rejected/failed episodes. **Do not
train** if many teacher episodes fail; inspect the cause. Initial collection
has `traffic_density=0`; this is not evidence about traffic safety.

2. Fit learner on episodes from only train seeds, evaluate offline on only val:

```powershell
python agent4_learned_residual_v33\train_student_v33.py --data-dir ".\agent4_v33_teacher_data" --model ".\pdd_v2_models\at_t\model_v2.joblib" --out-model ".\agent4_v33_student.joblib" --out-metrics ".\agent4_v33_training_metrics.json"
```

3. Evaluate **actual closed-loop** MetaDrive steering from the new model,
compared with original PDD baseline and V3.1 manual residual, on untouched
seeds 28/29 (scale to 28–31 only after pilot):

```powershell
python agent4_learned_residual_v33\eval_student_v33.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --student ".\agent4_v33_student.joblib" --maps C,X --seeds 28,29 --profiles baseline_3,lateral_3,student_v33 --steps 1400 --out ".\agent4_v33_closed_loop_eval.json"
```

## Interpretation / safeguards

- `teacher_action` / `teacher_steer_md` comes from MetaDrive IDM, **not** from
  PDD. Teacher success is **not** a learned-model success.
- Model steering is clipped to ±0.35, with the same V3.1 `max_steer_delta`.
  `out_of_road` from MetaDrive and lane/heading guards are unchanged.
- If teacher fails to complete C/X at **3 m/s** with injected steering pulses,
  reduce `--perturb-md` to `0`, or inspect the map; don't silently include a
  failure episode as successful supervision.
- Dataset splits are by **complete seed and map episode**; train/val must be
  disjoint, and eval seeds **28–31** cannot be included in either. Eval script
  checks for seed leakage. Previously tested seeds 0–3 are for diagnosis, not
  valid unseen test seeds.
- All model training is from new labelled simulation data, on top of the
  pre-existing frozen PDD steering model. It's an imitation-learning transfer
  experiment, NOT proof of PDD-only generalization.
- Offline MAE is a regression diagnostic only; **closed-loop success rate**
  on unseen C/X seeds is the actual experiment.
- Speed governor still supplies ALL longitudinal control. Neither original
  PDD throttle/brake nor the teacher IDM longitudinal action drives in eval.
- `road_margin_approx_m` is geometric approximation and does not replace the
  official MetaDrive line-contact/out-of-road predicate.
- No real-world car, camera-based autonomous driving, traffic handling, or
  roadworthiness is claimed.
