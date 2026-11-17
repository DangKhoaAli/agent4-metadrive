# Agent 4 V3.9.1 — Steering drift ablation (MetaDrive 0.4.3)

## Why this exists

V3.9 showed that V3.7 and V3.8 are blocked above 4 m/s with `Bad target speed`, **before taking any step**. Do not treat these as driving failures. Only the unchanged governor reached 6/8/10 m/s, and failed to complete long straight `SSSSS` even at 4 m/s. At 4 m/s it reached `lane_departure_guard` for seed 120 and `out_of_road` for seed 121. V3.8 also regressed on empty road (15/18 vs V3.7's 18/18).

**Goal:** isolate steering weaknesses *before* attempting high-speed longitudinal training.

## Structure

Keep directory `agent4_steering_diagnostic_v391` as sibling of previous `agent4_*` folders. Requires: `agent4_controller_v30`, `agent4_controller_v31`, `agent4_learned_residual_v33`, `agent4_robustness_v34`, `agent4_multispeed_v35`, and `pdd_v2_models/at_t/model_v2.joblib` and frozen `agent4_v35_student.joblib`.

Three steering modes, with **the same governor**, no traffic, no new training:

- `pdd_only`: original PDD V2 steering (baseline_3 internals)
- `rule_lateral`: PDD + the V3.1 rule-based lateral/heading correction (lateral_3)
- `learned_v35`: PDD + frozen learned V3.5 residual

If `rule_lateral` succeeds at high speed while `learned_v35` fails, the learned steering residual becomes the main candidate for retraining. If all fail, examine route alignment/actuator, road continuity and tracking horizon; don't immediately assume a model-data problem.

## Run (PowerShell in dataset root)

```powershell
python agent4_steering_diagnostic_v391\test_v391.py

python agent4_steering_diagnostic_v391\diagnose_v391.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --maps S,SSSSS --seeds 122,123 --speeds 4,6 --profiles pdd_only,rule_lateral,learned_v35 --steps 2500 --out ".\agent4_v391_steering_ablation.json"
```

This command executes `2 maps × 2 seeds × 2 speeds × 3 modes = 24` episodes. To reduce cost, use `--maps SSSSS` for 12 episodes. Only if the first test is encouraging, add `--speeds 8,10` on new seeds.

### Report interpretation

- `status=arrive_dest`: successful arrival.
- `status=lane_departure_guard`: planned lateral tracking error reached guard threshold 1.5 m; *not necessarily physical road departure*.
- `status=out_of_road`: MetaDrive environment reports out-of-road/line contact.
- `max_speed_mps`: actual max observed ego speed; target speed is not the same thing.
- `sim_elapsed_s`: simulated time, not wall-clock time.
- `initial_lane_index`: helps identify lane or map changes.

Training at higher speed remains necessary after diagnostics. Never disable safety/line-contact guards to create a misleading 'success'.

**Report files:** `agent4_v391_steering_ablation.json` and matching `.csv`. Send the JSON back for analysis.

## Limitations

Offline test verifies Python logic only. Full simulator verification must run on a computer with MetaDrive 0.4.3 and the original trained models. Rule correction is an AB diagnostic, **not a substitute learned Agent4 final policy**. Also no traffic or physical-car safety claims.
