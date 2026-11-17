# Agent 4 V3.6.1 — Diagnostic pha tăng tốc (KHÔNG huấn luyện lại)

## Tại sao có phép thử này?

- V3.6 đã huấn luyện model ga/phanh bằng 63.791 mẫu train và 25.789 validation, nhưng trên `C,X` seed `76,77`, learned controller chỉ về đích `8/12`, so với governor `10/12`.
- Ở `4 m/s`, learned controller `0/4`; trong cả 4 lần, xe chạm vạch vàng liền.
- Điểm khác biệt rõ rệt: learned throttle ban đầu `+1.0`, trong khi governor tối đa `+0.4`. Ví dụ C76: learned đạt ~4 m/s ở step 18–20, governor chỉ ~2,26 m/s ở step 20. Đó là **giả thuyết về nguyên nhân**, chưa chứng minh.

V3.6.1 không thay model, không thay steering V3.5, không chỉnh MetaDrive. Chỉ thêm một nhánh **kiểm tra đối chứng**:

1. `governor_v35`: tăng tốc theo quy tắc gốc.
2. `learned_long_v36`: ga/phanh của model V3.6 nguyên vẹn.
3. `learned_launch_cap_v361`: vẫn dùng đúng model V3.6 nhưng **trong 40 lần dự đoán đầu tiên, lệnh ga dương bị giới hạn tối đa +0.4**; lệnh phanh âm không bị can thiệp. Từ bước 40 trở đi, model áp dụng nguyên vẹn.

> **Cảnh báo:** Cap là một quy tắc do chúng ta thêm để xác định nguyên nhân; nếu xe thành công, **KHÔNG được gọi cấu hình này là Agent4 hoàn toàn do model học**. Phép thử cũng chưa chứng minh riêng sự khác biệt tăng tốc là nguyên nhân duy nhất.

## 1. Cài đặt

Giải nén thành `dataset\agent4_launch_diagnostic_v361` ngang hàng `agent4_longitudinal_v36`, `agent4_multispeed_v35`, `agent4_robustness_v34`, các module steering trước đó và file model `.joblib`.

## 2. Test

```powershell
python agent4_launch_diagnostic_v361\test_v361.py
```

## 3. Benchmark đối chứng đúng lỗi 4 m/s

```powershell
python agent4_launch_diagnostic_v361\benchmark_launch_v361.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --longitudinal ".\agent4_v36_longitudinal.joblib" --maps C,X --seeds 76,77 --speeds 4 --profiles governor_v35,learned_long_v36,learned_launch_cap_v361 --launch-steps 40 --launch-cap 0.4 --steps 1800 --trace-stride 10 --out ".\agent4_v361_launch_ablation.json"
```

12 episodes, cùng map / seed / tốc độ, trong đó bốn episode cap đối chứng mới. Không dùng IDM khi chạy, không huấn luyện mới.

## 4. Cần quan sát gì?

- `success_by_profile`: cap 4/4 hay tiếp tục thất bại?
- `episodes[].summary.speed_checkpoints_mps`: tốc độ tại step 20/40.
- `episodes[].summary.launch_diagnostic.capped_positive_calls`: model có thực sự bị giới hạn trong pha xuất phát?
- Các episode `out_of_road`: `terminal_reasons`, `travelled_approx_m`, `max_abs_planned_lateral_right_m`, `trace[].action_applied`.

Nếu cap cải thiện rõ rệt, ưu tiên **thu thập dữ liệu dưới steering V3.5 với đa dạng pha tăng tốc**, huấn luyện lại longitudinal bằng DAgger hoặc dataset aggregation, thay vì dùng cap thủ công lâu dài. Nếu cap không cải thiện, xem xét steering/longitudinal coupling ở 4 m/s và các trạng thái lệch làn.

**Giữ nguyên seed 78–79** như tập chưa đụng tới, không dùng chúng để lựa chọn cap, hệ số hay retrain.

## Giới hạn

Chưa chạy trên MetaDrive thực tế trong môi trường tạo mã. Đã kiểm tra cú pháp, test patch/restore giả lập. Không phải chứng nhận an toàn cho xe thật; không có giao thông, target speed vẫn do planner theo quy tắc.
