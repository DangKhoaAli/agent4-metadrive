# Agent 4 V3.0 — thử điều khiển ôm cua A/B (MetaDrive 0.4.3)

## Mục tiêu

Báo cáo V2.9 trên map C seed 0: xe bị `out_of_road` ở bước 274 vì chạm `on_yellow_continuous_line`. Độ lệch hướng tăng đến ~7,85° trong khi steering model gần như không thay đổi. V3.0 so sánh hai chế độ chạy cùng model, cùng seed và tốc độ:

- `baseline`: chuyển dấu steering từ CARLA PDD sang MetaDrive đúng như V2.7–V2.9, không có residual.
- `residual`: `steer = clip(-model_carla*gain + clip(k_lat*offset_right - k_heading*heading_error, ±residual_limit), ±steer_limit)`, sau đó giới hạn thay đổi steering giữa các bước.

`offset_right` là độ lệch so với tâm **làn được chọn dọc tuyến**, dương về bên phải. V3.0 tự kiểm chứng dấu này bằng hình học lane.position, không đoán dấu. `heading_error` là (hướng xe - hướng tiếp tuyến làn) chuẩn hóa về [-π, π], dương khi xe hướng lệch trái. Hệ số ban đầu `k_lat=0.035`, `k_heading=0.30`, residual tối đa `±0.08` **chỉ là giả thuyết kiểm thử**, không phải hệ số tối ưu hoặc được CARLA xác nhận.

**V3.0 không train lại model**, nên không được coi điều khiển residual là 100% machine learning. Model vẫn điều khiển steering chính, nhưng có thêm điều khiển quy tắc. Ga/phanh vẫn do bộ giữ tốc độ độc lập, không dùng IDM, không có giao thông. Chỉ dùng trong mô phỏng.

## Cài đặt

Giải nén `agent4_controller_v30/` bên trong `dataset/`, ngang hàng:

- `agent4_pdd_v2/`
- `agent4_calibration_v22/`
- `agent4_calibration_v23/`
- `agent4_calibration_v24/`
- `agent4_calibration_v27/`
- `agent4_diagnostic_v29/`
- `pdd_v2_models/at_t/model_v2.joblib`

Dùng **cùng môi trường Python 3.10**, MetaDrive 0.4.3, scikit-learn, joblib và numpy đã chạy V2.9. Không cần cài phiên bản package mới.

## Cách chạy

Từ thư mục `dataset`:

```powershell
python agent4_controller_v30\test_residual_v30.py

python agent4_controller_v30\ab_closed_loop_v30.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C --seeds 0 --steps 1200 --out ".\agent4_v30_AB_C.json"
```

Lệnh mặc định thử `baseline` trước rồi `residual`, cùng map C seed 0, tốc độ 3 m/s, gain 1. Baseline giữ nguyên action logic của V2.9 và các ngưỡng cơ bản; V3.0 thêm kiểm tra hình học/heading và logging nên chưa bảo đảm tái hiện từng số cuối cùng một cách tuyệt đối.

Lệnh có thêm guard mang tính **ước lượng hình học** trước khi đến gần mép đường (cả hai chế độ sử dụng chung guard, nên không giống nguyên bản V2.9):

```powershell
python agent4_controller_v30\ab_closed_loop_v30.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C --seeds 0 --steps 1200 --guard-road-margin-m 0.20 --out ".\agent4_v30_AB_C_guard.json"
```

`--guard-road-margin-m 0` (mặc định) tắt *guard ước lượng* để so sánh gần nhất với V2.9. Dù tắt, **điều kiện kết thúc `on_continuous_line_done` thực của MetaDrive vẫn giữ nguyên**. Độ lệch góc vượt 0.45 rad hoặc độ lệch tâm làn vượt 1.5 m dừng episode. Đây là thực nghiệm trong mô phỏng, không chứng minh an toàn thực tế.

## Đọc kết quả

- `status=arrive_dest`: xe tới đích; vẫn là thử điều khiển steering-only hybrid.
- `status=out_of_road`: điều kiện MetaDrive kết thúc, đọc `terminal_road_state.active_reasons`.
- `status=road_margin_guard`: dừng **trước khi** phạm vạch dựa trên ước lượng, không phải đã chạm vạch.
- `status=lane_departure_guard / heading_departure_guard`: dừng chủ động theo giới hạn.
- `status=observation_or_model_guard`: lỗi lấy tính năng/làn/mô hình; đọc `errors`.
- `max_abs_heading_error_rad`, `max_abs_planned_lateral_right_m`: độ lệch lớn nhất.
- `correction_active_steps`, `steering_saturation_steps`: kiểm tra residual có được áp dụng và có bị clipping không.
- `trace`: mỗi 5 bước và 25 bước cuối, hoặc khi gần rìa. `residual_components` tách 2 nguồn điều chỉnh.

**Chú ý**: Khoảng cách xa hơn chưa đủ chứng minh cải thiện. So sánh `arrive_dest`, số bước, va chạm, vạch liền, route completion, lệch hướng, lệch làn. Thử thêm seed 1,2 khi seed 0 có kết quả. **Không tăng gain hoặc bỏ điều kiện chạm vạch để che lỗi.**

Sau khi chạy, gửi `agent4_v30_AB_C.json`. Nếu residual cải thiện, bước sau mới đánh giá đa seed và lên kế hoạch tạo dữ liệu lệch trạng thái (DAgger/behavior cloning) để model tự học sửa sai thay vì phụ thuộc rule lâu dài.
