# Agent 4 — V3.2 Feature Audit for failures on C/X seed 2

**Purpose:** Diagnose why the same trained PDD steering model + bounded residual
fails on `C` seed 2 (continuous-yellow-line contact before CircularLane) and
`X` seed 2 (departure guard with a large heading error). Do not tune gains yet.

## Requirements

Keep sibling folders under your `dataset` directory:

- `agent4_feature_audit_v32` (this folder)
- `agent4_controller_v31`, `agent4_controller_v30`
- All existing V3.0 dependencies (`agent4_calibration_v27`, `agent4_diagnostic_v29`, and older adapters)
- `pdd_v2_models/at_t/model_v2.joblib`

Run from the `dataset` directory in PowerShell:

```powershell
python agent4_feature_audit_v32\test_feature_audit_v32.py

python agent4_feature_audit_v32\audit_features_v32.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C,X --seeds 2 --profiles baseline_3,lateral_3 --steps 1400 --trace-stride 1 --reference ".\agent4_v31_holdout_CX123.json" --out ".\agent4_v32_feature_audit_CX2.json"
```

Upload `agent4_v32_feature_audit_CX2.json` for assessment.

## What is captured

Each recorded **pre-action** step contains:

- `feature_audit.features_pdd_v2`: all 15 numerical inputs actually sent to the learned model, including both 25m/45m waypoints and nearby traffic fields
- `feature_audit.route_lane_indices`: planned lane chain chosen by existing V2.7 sampler
- `geometry_audit`: planned/current lane ids, lane tangent, local lateral sign
- `model_steer_md`, `residual_steer`, `action_applied`, lateral/heading errors, edge margin
- `raw_residual_unclipped`, `residual_limit_active`, `model_and_residual_opposed`
- `audit.audit_alignment_ok`: check that features align with applied actions
- `reference_comparison.near_exact_reproduction`: should be true if replay deterministic and action implementation unchanged

**No change in control:** same speed 3 m/s, gains, waypoint routine, guard thresholds,
MetaDrive line-contact termination and model as V3.1. There is no learned throttle/brake.
Logging more frequently may increase output size. Model is *not retrained*.

## Limitations

- `guard_road_margin_m=0`: consistent with V3.1, **not** a robust safety design
- Edge margin is an approximation and can be negative even without an out-of-road flag; rely on MetaDrive for authoritative episode status
- Seed 2 is a debugging set, not holdout evidence; tune/training decisions require separate seeds afterward
- Only use inside MetaDrive simulation. Do not use this controller on a physical vehicle
