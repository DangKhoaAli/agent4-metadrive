# Agent 4 V3.8 — Controlled-lead traffic behavior cloning (MetaDrive)

## Scope and interpretation

This is **an initial traffic-following research experiment**, not a validated autonomous driver.

**V3.8 spawn fix (2026-11-16):** The initial 10 m road segment is too short for a 24 m lead gap. `resolve_lead_spawn_slot()` now follows the ego navigation checkpoints across connected road segments and spawns on the correct downstream lane. Lead speed scheduling also now uses an explicit longitudinal action, because IDM can overwrite its internal `target_speed`; steering still comes from IDM. The smoke test still must pass on your MetaDrive 0.4.3 installation.

- **Steering:** unchanged frozen CARLA PDD V2 + learned MetaDrive Student V3.5.
- **Longitudinal (baseline):** learned V3.7, trained to imitate the empty-road governor.
- **Longitudinal (V3.8):** newly trained model imitating an explicitly **hand-designed gap-aware teacher** driving with steering V3.5.
- **Traffic:** one scripted, physically spawned lead car in the ego lane. Native random traffic density is 0 to isolate the controlled experiment.
- **Scenarios:** `slow_release` (1 m/s → 5 m/s), `stop_release` (stopped → 5 m/s), `brake_release` (2 m/s → stopped → 5 m/s). The leading car is driven by an IDM lane-following policy with a scripted speed schedule. This is **not** a naturalistic-IDM traffic dataset.
- **Initial map:** `S` only for deterministic same-route lead placement across consecutive road segments. Lanes, vehicle state and route are obtained from privileged MetaDrive simulation state.
- **Unchanged rules:** route/target-speed planner, steering guards. Model V3.8 acts directly with **no fallback** to teacher or governor during evaluation.

### Required sibling folders and files

Extract `agent4_traffic_v38` **inside the same `dataset` root** as all earlier agent folders. Requires `agent4_controller_v30`, `agent4_calibration_v27`, `agent4_diagnostic_v29`, `agent4_robustness_v34`, `agent4_learned_residual_v33`, `agent4_multispeed_v35`, `agent4_longitudinal_v36`, and `agent4_onpolicy_longitudinal_v37`. Requires trained `.joblib` files **on your computer** (not packaged).

Python 3.10, MetaDrive 0.4.3, numpy, joblib, scikit-learn.

### 1. Offline tests

```powershell
python agent4_traffic_v38\test_v38.py
```

### 2. **MetaDrive API smoke test — must pass before collecting data**

```powershell
python agent4_traffic_v38\smoke_v38.py --map S --seed 128 --gap 24 --steps 15
```

The code creates a real `TrafficDefaultVehicle`, registers its traffic policy in MetaDrive `TrafficManager`, and checks that a vehicle is detected in the correct lane. **If this fails, do not start training.** The code was developed against publicly documented MetaDrive interfaces, but execution under your exact local 0.4.3 installation must be confirmed.

### 3. Collect actual traffic interactions (teacher demonstrations)

```powershell
python agent4_traffic_v38\collect_v38.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --maps S --speeds 2,3,4 --scenarios slow_release,stop_release,brake_release --train-seeds 100-111 --val-seeds 112-115 --eval-seeds 116-119 --gap 24 --steps 1600 --out-dir ".\agent4_v38_traffic_data"
```

This accepts **only** teacher episodes that reach destination **and** have actual front-car observations, close following, rule-teacher intervention and braking. Failed episodes remain in the manifest with `accepted=false`; they are not silently counted as training examples. Inspect `manifest.json` before training.

### 4. Train on the observed state → actually applied teacher action

```powershell
python agent4_traffic_v38\train_v38.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --data-dir ".\agent4_v38_traffic_data" --out-model ".\agent4_v38_traffic.joblib" --out-metrics ".\agent4_v38_training_metrics.json"
```

Input = all V3.7 longitudinal features **plus** `{front_present, front_gap_m, front_speed_mps, front_closing_mps, front_ttc_s}`. Labels are actual traffic-teacher throttle/brake. Data split by **seed**, no overlap with V3.7's 80–99.

### 5. Closed-loop comparison on untouched seeds (before inspecting errors)

```powershell
python agent4_traffic_v38\eval_v38.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --old-longitudinal ".\agent4_v37_longitudinal.joblib" --longitudinal ".\agent4_v38_traffic.joblib" --maps S --seeds 116,117 --speeds 2,3,4 --scenarios slow_release,stop_release,brake_release --profiles teacher_v38,learned_v37,learned_v38 --gap 24 --steps 1600 --out ".\agent4_v38_traffic_eval.json"
```

**54 episodes** (2 seeds × 3 speeds × 3 scenarios × 3 profiles), with *identical* steering, scripted traffic and seed. Any episode with a crash, line departure, step limit, or guard stop is **not** an arrival. Compare: `arrive_dest`, `crash`, minimum bumper gap, time-to-collision (only when closing), following brake counts, and lane-clearance warnings.

If this appears promising, run **fresh reserved evaluation seeds 118,119** without retraining. Do not move evaluation seeds into training.

## Limitations and troubleshooting

1. **API stability:** TrafficManager internals for injecting a controlled lead (`_traffic_vehicles`, `spawn_object`, `add_policy`) may vary across MetaDrive versions. Verify the smoke test first; no MetaDrive simulator is installed in the assistant environment.
2. **Geometry:** Lead placement traverses `ego.navigation.checkpoints` and `road_network.graph`, so a 24 m gap may land on a lane after the first 10 m starting segment. If the route itself is too short or lane count changes, the experiment fails rather than silently choosing a wrong lane. Requested gap is centre-to-centre; `observe_front` returns bumper-to-bumper gap (normally ~4–5 m smaller).
3. **Observability:** The lead is detected by searching actual registered traffic vehicles ahead, filtering relative lateral offset and heading. If front encounters are rare or not verified, collection deliberately errors rather than generating misleading traffic data.
4. **Behavior cloning limitations:** The teacher is a *rule-based headway controller*; this teaches imitation, not automatic reward maximization. Offline MAE does not prove following performance in closed loop.
5. **Safety:** These experiments are confined to MetaDrive simulation and cannot establish real-world driving safety.

## Required reports to send back

`agent4_v38_traffic_data/manifest.json`, `agent4_v38_training_metrics.json`, `agent4_v38_traffic_eval.json`. Also send smoke-test terminal output if it fails.
