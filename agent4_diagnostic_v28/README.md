# Agent 4 V2.8 — Diagnose V2.7 curve failure

**Purpose**: Investigate why V2.7 returned `out_of_road` on MetaDrive map `C`, seed `0`, despite a small recorded nearest-lane offset. This script deliberately **uses exactly the same control law as V2.7**: PDD V2 model controls steering; independent governor controls acceleration/braking. It does not use IDM, modify model weights, or increase steering gain.

## Files
- `diagnose_curve_v28.py`: diagnostic pilot with per-step logging.
- `test_diagnostic_v28.py`: small math/format tests.

## Installation
Extract `agent4_diagnostic_v28` next to existing `agent4_calibration_v27` and all earlier V2 adapter folders in `dataset`. Uses your existing Python 3.10 / MetaDrive 0.4.3 environment and existing `model_v2.joblib`.

## Run

```powershell
cd "D:\Học\Tri tue nhan to nang cao\lab\dataset"
python agent4_diagnostic_v28\test_diagnostic_v28.py
python agent4_diagnostic_v28\diagnose_curve_v28.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C --seeds 0 --steps 1200 --out ".\agent4_v28_diagnostic_C.json"
```

After checking the same 3 m/s run, an **optional independent speed sensitivity test** is:

```powershell
python agent4_diagnostic_v28\diagnose_curve_v28.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C --seeds 0 --steps 1600 --speed-limit-mps 2 --out ".\agent4_v28_diagnostic_C_2mps.json"
```

## Interpretation
Compare:
- `status`, `n_applied_steps`, `travelled_approx_m` against V2.7 (`out_of_road`, 274, about 76.75 m). Different results warrant investigation of nondeterminism.
- `trace_every_step[*].before.lane_index` and `after.lane_index`: nearest/current lane reassignment.
- `before.lane_lateral_m` vs `before_plan.planned_lane_lateral_m`: lane-local offset vs planned-lane offset. **Neither is the actual road-edge clearance.**
- `before.heading_error_rad`: orientation relative to current lane tangent.
- `sim_info.out_of_road`, other flags at termination.
- `model_wp1_right_m`, `model_wp2_right_m`, applied steering: whether the learned response changes as geometry changes.

**Do not** interpret an `out_of_road` result with `lane_lateral_m` smaller than 1.5 m as inconsistent simulator behavior: MetaDrive can use road geometry/vehicle shape different from nearest lane-center offset. This tool logs additional evidence but may not expose the precise collision boundary.

### Safety/limits
Traffic-free simulator only. Steering predicted by model; throttle/brake is **not learned**. Same experimental steering gain=1.0, steering limit=0.35 and max change=0.12 as V2.7. No test on a real car. Do not apply gain changes derived from IDM regression without further testing.
