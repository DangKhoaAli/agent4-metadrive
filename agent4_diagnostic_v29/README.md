# Agent 4 V2.9 — chính xác điều kiện MetaDrive `out_of_road`

Chẩn đoán bổ sung từ V2.8, **không thay model hoặc action**. Không chạy IDM, không dùng longitudinal model, không thay gain.

MetaDrive 0.4.3: `_is_out_of_road()` xét `not vehicle.on_lane` hoặc (khi bật `on_continuous_line_done` và không bật `out_of_route_done`) các cờ `on_yellow_continuous_line`, `on_white_continuous_line`, `crash_sidewalk`. `on_broken_line_done` có thể bổ sung điều kiện riêng.

## Cài đặt
Giải nén `agent4_diagnostic_v29` vào thư mục `dataset`, cùng cấp `agent4_diagnostic_v28`, `agent4_calibration_v27`.

```powershell
python agent4_diagnostic_v29\test_road_flags_v29.py
python agent4_diagnostic_v29\diagnose_road_flags_v29.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C --seeds 0 --steps 1200 --out ".\agent4_v29_road_flags_C.json"
```

Xem `terminal_road_state.active_reasons`, `terminal_road_state.flags`, `terminal_road_state.termination_config`, `terminal_road_state_matches_info`. Giá trị `active_reasons=[]` chỉ có ý nghĩa nếu các trường liên quan có dữ liệu, xem `predicted_out_of_road`.

**Không tắt các cờ kết thúc**, vì đây là phép kiểm chứng an toàn trong mô phỏng. Mọi phép thử chỉ dùng MetaDrive 0.4.3 không giao thông, không xác nhận an toàn đường thật.

Mã được kiểm tra cú pháp và unit test, **chưa được chạy trong MetaDrive trên Windows của người dùng**.
