# V3.12B changes

- Full-step (0.1 sec) raw and verified applied IDM action capture.
- Explicit current_action vs vehicle control vs clipped raw action verification.
- Never use raw overrange actions as directly supervised labels.
- Planned-lane pose and 5/10/20 m heading lookahead, matching connected lane geometrically.
- Episode-level safety gates and sample-level anomaly flags.
- Atomic per-episode CSV and manifest writes; resumable by exact configuration.
- Disjoint seed train/val/test splits; offline dataset validation.
- No joblib training, no model changes, no traffic or real-world claims.
