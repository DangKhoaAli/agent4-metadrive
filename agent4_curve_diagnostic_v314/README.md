# Agent 4 V3.14 — Closed-loop curve diagnostic (read-only)

## Purpose
The current ExtraTrees steering policy on MetaDrive has high offline fidelity but may drift at selected curves. This package audits JSON from V3.13 **without changing the model**. Python standard library only.

## Run (from Windows `dataset/` folder)

```powershell
python agent4_curve_diagnostic_v314\test_v314.py
python agent4_curve_diagnostic_v314\analyze_v314.py --test ".\agent4_v313_closed_loop_test.json" --metrics ".\agent4_v313_model\metrics_v313.json" --val ".\agent4_v313_val_smoke.json" --out ".\agent4_v314_diagnostic.json"
```

Writes `agent4_v314_diagnostic.json` and `.csv`.

## Validation BEFORE model changes

```powershell
python agent4_steering_v313\eval_closed_loop_v313.py --model ".\agent4_v313_model\agent4_v313_steering.joblib" --maps C --seeds 186-189 --speeds 2,3,4 --trace-stride 1 --out ".\agent4_v314_C_val_fulltrace.json"

python agent4_curve_diagnostic_v314\analyze_v314.py --test ".\agent4_v314_C_val_fulltrace.json" --out ".\agent4_v314_C_val_audit.json"
```

- Existing **test seeds 190–193 have already been inspected**. Do not use them to select or tune any next-generation model. Evaluate candidate changes with **validation seeds 186–189** and, after freezing the model, measure generalization on **new, untouched final seeds 194–201**.
- Run fixed `--trace-stride 1` to investigate abrupt changes in `planned_lateral_right_m`, `is_circular_lane`, heading and predicted steering.
- `lateral_vs_clearance_proxy_disagreement` means **different geometry signals disagree**; it is NOT proof of safe driving or proof of an implementation bug. Verify `planned_lane_id`, `vehicle_lane_id`, left/right side distances with a full-step diagnostic if present.
- `out_of_road` comes from MetaDrive; clearance is a conservative **proxy** and not a safety claim.
- All steering policies are model-only: no IDM teacher fallback, but throttle remains rule-based, and geometry remains privileged simulator information.

## Observed V3.13 baseline
- 36 independent simulation combinations on test-seeds 190–193; 33 arrived, 31 quality accepted.
- `C seed190`: failed 2, 3, 4 m/s, actual `out_of_road`.
- `C seed192`: arrived but quality failed at 2 and 3 m/s; lateral proxy up to 1.74m despite large clearance proxy. Investigate reference-lane switching and model's off-expert-distribution behavior; do not assume root cause.

## What is NOT implemented
No retraining, no corrective policy, no data augmentation and no driving safety guarantees. Next model iteration should be judged by **closed-loop quality on validation first**, not offline MAE alone.

## V3.14 detailed geometry capture (NEW)

V3.14 also provides `eval_diagnostic_v314.py`, a copy of the frozen V3.13 evaluator that adds **per-step lane identity and geometry** to trace without changing the model, its features, predictions, or longitudinal controller. It must be placed in the `dataset/` directory next to V3.13 and V3.12B.

```powershell
python agent4_curve_diagnostic_v314\eval_diagnostic_v314.py --model ".\agent4_v313_model\agent4_v313_steering.joblib" --maps C --seeds 186-189 --speeds 2,3,4 --trace-stride 1 --out ".\agent4_v314_C_validation_geometry.json"

python agent4_curve_diagnostic_v314\analyze_v314.py --test ".\agent4_v314_C_validation_geometry.json" --out ".\agent4_v314_C_validation_geometry_audit.json"
```

New trace fields: `planned_lane_id`, `vehicle_lane_id`, their lane classes, `left_side_m`, `right_side_m`, `distance_to_lane_end_m`, `selected_next_lane_id`, `selected_next_lane_seam_m` and lookahead availability/headings. A sudden lateral change that coincides with `planned_lane_id` switch is a strong **diagnostic lead**, not by itself a correctness proof. For high-confidence reproduction of the already-observed test issue, rerun `--seeds 192 --speeds 2,3` **for diagnosis only**; never use that test seed to fit/tune the next model.
