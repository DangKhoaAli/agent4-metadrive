# V3.13 (2026-11-17)

- Introduces offline IDM imitation training on full-step verified V3.12B CSVs; never trains from manifest metadata alone.
- 12-feature frozen schema, explicit horizon availability and safe numeric checks.
- 3 candidates: Ridge, HistGradientBoostingRegressor, ExtraTreesRegressor, equal episode weights.
- Validation-only model selection, untouched test result, map/speed/lane-specific MAE/RMSE/P95.
- No train/val/test seed leakage; rejects missing CSV, malformed labels, unapproved/partial episode.
- Independent closed-loop MetaDrive evaluation: model steering only, separate fixed P longitudinal governor, no IDM or teacher fallback, quality criteria aligned to V3.12B.
- Offline tests: 9 PASS, including synthetic end-to-end fit/checkpoint/load and fake-environment applied-action alignment.
- Training on real dataset and actual MetaDrive closed-loop validation **not run** in artifact authoring environment.
