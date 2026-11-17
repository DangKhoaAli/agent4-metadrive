# Agent 4 V3.1: kiểm tra tham số hiệu chỉnh và ảnh hưởng tốc độ

## Bối cảnh

Báo cáo `agent4_v30_AB_C.json` (C, seed 0, 3 m/s): baseline 274 bước,
76,75 m, 33,99% tuyến; residual 286 bước, 80,35 m, 35,37% tuyến.
**Cả hai thất bại** do `on_yellow_continuous_line`; V3.0 residual giúp giảm
lệch hướng/lệch làn nhưng chưa hoàn thành tuyến.

V3.1 **không huấn luyện lại** model PDD và **không thay đổi** cách lấy waypoint.
Nó chạy lại `pilot_episode` của V3.0 trên các nhóm tham số đã định trước,
không sử dụng IDM và không cho model điều khiển ga/phanh.

## Cài đặt

Giải nén `agent4_controller_v31/` vào `dataset/` (ngang hàng
`agent4_controller_v30/`, `agent4_calibration_v27/`,
`agent4_diagnostic_v29/`, `pdd_v2_models/`). Giữ đầy đủ các folder.

## Unit test

```powershell
python agent4_controller_v31\test_profiles_v31.py
```

Expected: `Agent 4 V3.1 profile and steering tests: PASS`.

## A/B thử ban đầu (seed 0)

```powershell
python agent4_controller_v31\sweep_closed_loop_v31.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C --seeds 0 --steps 1400 --out ".\agent4_v31_sweep_C0.json"
```

Mặc định thử 6 profile, tất cả ở cùng map/seed:

| Profile | Tốc độ (m/s) | Kp ngang | Kp hướng | Giới hạn residual | Ý nghĩa |
|---|---:|---:|---:|---:|---|
| baseline_3 | 3 | n/a | n/a | n/a | Mốc model steering gốc |
| v30_3 | 3 | 0.035 | 0.30 | 0.08 | Lặp lại V3.0 residual |
| heading_3 | 3 | 0.035 | 0.60 | 0.08 | Chỉ tăng hiệu chỉnh góc hướng |
| lateral_3 | 3 | 0.080 | 0.30 | 0.08 | Chỉ tăng hiệu chỉnh lệch ngang |
| baseline_2 | 2 | n/a | n/a | n/a | Isolate tác động giảm tốc |
| v30_2 | 2 | 0.035 | 0.30 | 0.08 | Residual cũ tại tốc độ thấp |

**Lưu ý:** đây là thiết kế thử nghiệm, không phải các hệ số đã hiệu chuẩn.
Có thể chạy `balanced_3` riêng sau khi phân tích kết quả đầu tiên.

### Tùy chọn kiểm tra thêm

Nếu ít nhất một profile có triển vọng, kiểm chứng trên các seed khác,
**không tinh chỉnh theo seed 1/2 rồi gọi chúng là phép thử độc lập**:

```powershell
python agent4_controller_v31\sweep_closed_loop_v31.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C --seeds 1,2 --profiles "baseline_3,v30_3,heading_3,lateral_3" --steps 1400 --out ".\agent4_v31_holdout_C12.json"
```

## Đọc báo cáo

- `status=arrive_dest` **và** `route_completion` mới là tiêu chí hoàn thành.
- `step_limit`, `road_margin_guard`, `out_of_road`, `crash` KHÔNG phải thành công.
- `max_abs_heading_error_rad` đo hướng; `max_abs_planned_lateral_right_m`
  đo lệch ngang; `near_edge_warning_steps` là số lần khoảng hở *ước lượng*
  dưới 0,35 m (không phải số lần chạm vạch).
- `on_yellow_continuous_line` vẫn được giữ nguyên, không thay đổi điều kiện
  kết thúc nhằm tạo kết quả ảo.

## Giới hạn

- Chỉ MetaDrive, không giao thông, mô phỏng ở tốc độ thấp.
- Sử dụng action steering từ model học PDD + **phản hồi thủ công** ở profile residual.
- Ga/phanh dùng speed governor độc lập; chưa đánh giá policy điều khiển hoàn toàn.
- Không áp dụng bộ này lên xe vật lý.
- Đừng kết luận tổng quát từ một seed; cần nhiều seed/map và thử đường cong/giao lộ.
