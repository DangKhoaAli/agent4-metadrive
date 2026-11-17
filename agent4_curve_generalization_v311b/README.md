# Agent 4 V3.11B — Frozen V3.10 curved-road generalization

**Mục đích:** kiểm thử model V3.10 *đã huấn luyện* trên bản đồ cong `C`, `X`, **không huấn luyện lại** và **không thay trọng số**. Dùng chung PDD V2 và speed governor như V3.10; so sánh `pdd_only`, `rule_lateral`, `learned_v310` trên cùng map/seed/tốc độ.

## Sắp xếp thư mục

Giải nén `agent4_curve_generalization_v311b` vào **cùng thư mục `dataset`** với:

- `agent4_highspeed_training_v310` (bao gồm `eval_v310.py`, `common_v310.py`)
- `agent4_learned_residual_v33`
- `agent4_controller_v30`, `agent4_controller_v31`
- Các phụ thuộc đã dùng để chạy V3.10: `agent4_calibration_v27`, `agent4_diagnostic_v29` và phụ thuộc sâu hơn
- `pdd_v2_models/at_t/model_v2.joblib`
- `agent4_v310_steering.joblib`

**Python:** cùng môi trường Python 3.10, MetaDrive, joblib, numpy, sklearn từng chạy V3.10.

## Bước 1: test offline

```powershell
python agent4_curve_generalization_v311b\test_v311b.py
```

**PASS chỉ kiểm tra code xử lý dữ liệu/khôi phục, không chứng minh MetaDrive chạy được.**

## Bước 2: smoke test đường cong

```powershell
python agent4_curve_generalization_v311b\curve_eval_v311b.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v310_steering.joblib" --maps C,X --seeds 158 --speeds 2 --profiles pdd_only,rule_lateral,learned_v310 --steps 2500 --out ".\agent4_v311b_curve_smoke.json"
```

6 lượt (2 maps × 1 seed × 1 speed × 3 profiles). Kiểm tra status `arrive_dest`, `error_count`, `model_predictions_valid`, `near_edge_warning_steps` và `min_estimated_edge_clearance_m` trước khi chạy toàn bộ.

## Bước 3: đánh giá chính thức đường cong

```powershell
python agent4_curve_generalization_v311b\curve_eval_v311b.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v310_steering.joblib" --maps C,X --seeds 158-161 --speeds 2,3,4 --profiles pdd_only,rule_lateral,learned_v310 --steps 2500 --out ".\agent4_v311b_curves.json"
```

72 lượt (2 maps × 4 seeds × 3 speeds × 3 profiles). Kết quả **JSON và CSV**; report được lưu **sau từng lượt**, giúp không mất những lượt hoàn tất trước đó.

Nếu dừng và cần tiếp tục, chạy **chính lệnh bước 3 thêm `--resume` ở cuối**. Chỉ resume nếu toàn bộ cấu hình và SHA256 của hai model trùng. Nếu file đã tồn tại và muốn tạo mới từ đầu, dùng output mới hoặc `--overwrite` để cho phép ghi đè rõ ràng.

## Đọc kết quả

- `episodes[].summary.status`: `arrive_dest` là về đích; `out_of_road` là MetaDrive xác nhận ra đường; `lane_departure_guard` là ngưỡng chẩn đoán, **không nhất thiết thực sự ra đường**; `observation_or_model_guard` nghĩa lỗi đo/logic, không phải kết luận chất lượng steering.
- `episodes[].summary.confirmed_success`: `arrive_dest` và **không lỗi**; riêng student phải có prediction counts khớp số bước, không teacher fallback.
- `group_scores`: thành công theo map/tốc độ/controller, cảnh báo sát mép, lateral tối đa và margin ước lượng.
- `paired_comparisons`: hiệu số lateral/margin giữa student và PDD hoặc teacher trong **cùng map, seed, tốc độ**.
- `out_of_v310_training_speed=true` ở 2/3 m/s (V3.10 đã train target 4/6/8/10); **mọi thử C/X** đều là ngoại phân phối vì train trên `S`, `SSSSS`.
- Không tự suy diễn rằng đi hết đường là có khả năng xử lý giao thông: bài này **không có xe khác**, ga/phanh vẫn do governor quy tắc điều khiển.

**Quyết định sau khi có JSON:** nếu giáo viên thành công nhưng student thất bại => thu thập dữ liệu `C/X` để huấn luyện steering mới; nếu cả 3 thất bại cùng đoạn => kiểm tra lane chain, navigation geometry, đường và target speed trước. Không tăng tốc trên đường cong khi chưa qua 2–4 m/s.

## Scoring correction (V3.11B patch, 2026-11-17)

The first V3.11B scorer erroneously required `summary.measured_steps_aligned`,
which `eval_v310.run_one` never supplies. This marked every learned V3.10
rollout invalid even when all its predictions were applied. The patched scorer
checks `student_prediction_calls == n_applied_steps`, `student_predictions_aligned`,
no teacher fallback, no errors, and recorded action attribution consistency.
It does not modify student weights or the simulated actions.

Re-score the already recorded 3 m/s report **without rerunning MetaDrive**:

```powershell
python agent4_curve_generalization_v311b\rescore_v311b.py --input ".\agent4_v311b_curve_3mps.json" --out ".\agent4_v311b_curve_3mps_rescored.json"
```

For the old 2 m/s smoke report:

```powershell
python agent4_curve_generalization_v311b\rescore_v311b.py --input ".\agent4_v311b_curve_smoke.json" --out ".\agent4_v311b_curve_smoke_rescored.json"
```

`--out` must be a different path unless `--overwrite` is explicitly specified.
