# Agent 4 V3.15 — Recovery DAgger (additive, no legacy edits)

**Status:** New V3.15 implementation, offline unit-tested. **Not yet trained or closed-loop validated on your Windows MetaDrive.** V3.13 remains the baseline unless the closed-loop validation acceptance gate passes.

## What this does

- V3.13 ExtraTrees is **frozen**. New V3.15 learns a gated steering residual to improve recovery from deviations on `C` and `X`.
- A learner-controlled `MetaDriveEnv` introduces **small, deterministic steering pulses** only on V3.12B **TRAIN seeds**. A separately instantiated `IDMPolicy` merely supplies *shadow labels*; **the IDM policy is never assigned to `env.agent_policy` or used in `env.step`.**
- Raw IDM out-of-range steering, missing lookahead and unsafe geometry rows are discarded from training. Failed rollouts remain in diagnostics; usable pre-failure state labels can be retained if quality checks pass.
- Original TRAIN expert demonstrations give **anchor labels** so the new residual does not unnecessarily alter nominal steering.
- The new model is `baseline + alpha * recovery_gate(state) * clipped_residual_prediction`, and falls back ONLY to the **learned V3.13 baseline**, never to IDM. Longitudinal control remains rule-based.
- Validation compares V3.15 against V3.13 on **the exact same seed/map/speed pairs** and accepts V3.15 only if quality passes increase, no baseline-quality episode regresses, and no previously arrived episode becomes a failure.
- Final evaluation uses **new seeds 194–201** after freezing the candidate. Previously inspected test seeds 190–193 must never be used for tuning.

## One-time layout on your Windows machine

The GitHub repo was initially empty; **it does not contain the old code**. Keep existing folders unchanged in the parent `dataset` directory:

```text
D:\...\dataset\
├── agent4_idm_dataset_v312b\      # legacy code, unchanged
├── agent4_steering_v313\           # legacy code, unchanged
├── agent4_v312b_idm_dataset\       # existing full CSV dataset
├── agent4_v313_model\              # existing trained baseline model
└── agent4-metadrive\               # this GitHub repo
    └── agent4_recovery_v315\       # new code only
```

Clone the repository into `dataset` (or `git pull` if already cloned). If V3.15 is on a feature branch, switch to `feature/agent4-v315-recovery` for review; if merged, simply use `main`.

Install dependencies in the same Python environment that already runs V3.13: `metadrive==0.4.3`, `scikit-learn`, `numpy`, `joblib`.

## Run — only one PowerShell command

From inside `dataset\agent4-metadrive`:

```powershell
powershell -ExecutionPolicy Bypass -File .\agent4_recovery_v315\run_v315.ps1 -Stage all -PublishSummary
```

This runs unit tests, 96 training on-policy rollout combinations (`2 maps × 16 seeds × 3 speeds`), residual training, and **validation comparison** (`2 profiles × 2 maps × 4 seeds × 3 speeds = 48 episodes`). It writes the short, reviewable `reports/v315_validation_summary.json` **without uploading raw traces, model weights or full dataset**.

To quickly verify the environment before the full collection, run:

```powershell
python .\agent4_recovery_v315\test_v315.py
python .\agent4_recovery_v315\collect_v315.py --legacy-root .. --dataset ..\agent4_v312b_idm_dataset --baseline ..\agent4_v313_model\agent4_v313_steering.joblib --maps C --seeds 170 --speeds 3 --out-dir ..\agent4_v315_smoke
```

**Note:** Run `-Stage all` only after you are satisfied with the smoke results. `-Stage all` will resume collection and evaluation when matching output files exist. Training outputs are never silently overwritten.

## Promotion policy

The acceptance decision appears at `promotion.accepted` in `agent4_v315_validation.json`. `true` means V3.15 beat the baseline **on validation only**. It is **not** final proof of generalization.

After freezing any validated model, run final evaluation **once** on new holdout seeds:

```powershell
python .\agent4_recovery_v315\eval_v315.py --legacy-root .. --dataset ..\agent4_v312b_idm_dataset --model ..\agent4_v315_model\agent4_v315_recovery.joblib --mode final --confirm-frozen --maps S,C,X --seeds 194-201 --speeds 2,3,4 --out ..\agent4_v315_final_holdout.json
```

At no point does the evaluator instantiate IDMPolicy for steering; throttle is not learned and privileged simulator lane geometry is required. Results must not be represented as real-world autonomy or safety guarantees.

## Sharing results with ChatGPT

After `-PublishSummary`, inspect `reports/v315_validation_summary.json`, then push only that compact file:

```powershell
git add reports/v315_validation_summary.json
git commit -m "V3.15 validation result"
git push
```

Do **not** commit `agent4_v315_data`, `agent4_v315_model`, raw traces, CSVs, `.joblib` weights or large personal/secret files.
