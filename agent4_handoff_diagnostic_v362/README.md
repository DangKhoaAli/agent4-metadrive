# Agent 4 V3.6.2 — Exact Governor Handoff Diagnostic

**Purpose:** The V3.6.1 `clip(+0.4)` experiment did not reproduce the governor's full launch ramp. In fact, at C76 step 30, capped V3.6 used +0.4 while the governor used ~+0.205. This new experiment copies the **exact governor output** at each step of the initial interval, then switches exclusively to the already trained V3.6 model.

**This is NOT a new trained model.** It is a diagnostic intervention to isolate whether the learned controller's weakness is confined to starting acceleration or persists during steady driving. A success with an initial governor segment does NOT constitute an end-to-end learned policy.

## Requirements

Extract this entire directory alongside `agent4_longitudinal_v36`, `agent4_multispeed_v35`, `agent4_robustness_v34`, and all their previous dependencies. Use the same Python environment as the prior tests (MetaDrive 0.4.3, joblib, sklearn).

From your `dataset` PowerShell directory:

```powershell
python agent4_handoff_diagnostic_v362\test_v362.py

python agent4_handoff_diagnostic_v362\benchmark_handoff_v362.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --longitudinal ".\agent4_v36_longitudinal.joblib" --maps C,X --seeds 76,77 --speeds 4 --profiles governor_v35,learned_long_v36,handoff_80,handoff_160 --steps 1800 --trace-stride 10 --out ".\agent4_v362_exact_handoff.json"
```

16 runs total (4 cases × 4 controllers). Profiles:

- `governor_v35`: unchanged V3.5 steering and governor driving longitudinal throughout.
- `learned_long_v36`: unchanged V3.5 steering and V3.6 trained longitudinal throughout.
- `handoff_80`: **exact** governor action during steps 0..79, then raw V3.6 learned action from step 80 onward.
- `handoff_160`: **exact** governor action during steps 0..159, then raw V3.6 learned action from step 160 onward.

For hybrid modes the learned action is also computed in shadow (not applied) during the reference period. Steering model, simulator, termination rules and target-speed planner remain unchanged. There is no IDM policy in evaluation. Full model-control output will be logged in `trace[*].longitudinal_attribution_v362`, with active-source flags. `summary.governor_reference_applied_steps` and `summary.learned_longitudinal_applied_steps` disambiguate manual vs learned steps.

## Reading the diagnostic result

- If `handoff_80` succeeds: launch behavior was a major factor in these cases, but it doesn't prove it is the *only* issue.
- If `handoff_80` fails and `handoff_160` succeeds: a longer matching segment near curved segments matters.
- If even `handoff_160` fails while `governor_v35` succeeds: divergence is not confined to the first 160 steps; inspect longitudinal behavior after the switch and the steering interaction, and collect on-policy recovery data before training V3.7.
- Check `longitudinal_actions_aligned` and `error_count` before interpreting performance.

**Caveat:** Seeds 76/77 were already used in V3.6 diagnostics. Reserve seeds 78/79 as the final evaluation; do NOT use these to adjust any hyperparameters prior to final model freeze. Approximate road-margin does not replace actual MetaDrive road flags. No traffic; simulator privileged features; not physical safety validation.

Only offline unit tests, compilation, and ZIP integrity are checked here. MetaDrive closed-loop must run on your Windows machine.
