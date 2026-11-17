# Agent 4 V2.7 — Guarded learned-steering closed-loop pilot

## Mục tiêu

V2.6 xác nhận chiều steering đa phần đúng nhưng độ mạnh nhỏ hơn IDM ở cua/giao lộ; phép so sánh không đủ để gán gain. Bước V2.7 cho **Model PDD điều khiển steering thực sự** trong MetaDrive trên một bản đồ thử, nhưng **bộ giữ tốc độ độc lập** điều khiển ga/phanh. Đây là thử nghiệm *hybrid steering-only*, tuyệt đối không tính là model tự lái end-to-end.

Model V2 và trọng số không bị sửa. **Không dùng IDMPolicy** và không chuyển sang IDM khi thiếu waypoint. Không có giao thông. Tốc độ được hạn chế, steering bị giới hạn, có guard lệch làn. Kết quả dùng để tìm lỗi closed-loop, KHÔNG chứng minh xe đã tự hành hoàn chỉnh.

## Cài đặt

Giải nén thư mục `agent4_calibration_v27/` vào `dataset/`, ngang hàng với:

- `agent4_pdd_v2/`
- `agent4_calibration_v22/`
- `agent4_calibration_v23/`
- `agent4_calibration_v24/`
- `pdd_v2_models/at_t/model_v2.joblib`

Python 3.10, MetaDrive 0.4.3, numpy, joblib, sklearn dùng đúng environment đã chạy V2.6.

## Chạy từ thư mục dataset

```powershell
python agent4_calibration_v27\test_guarded_v27.py

# Pilot một lượt, đường thẳng trước
python agent4_calibration_v27\guarded_pilot_v27.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps S --seeds 0 --steps 650 --out ".\agent4_v27_guarded_S.json"
```

Chỉ sau khi xem kết quả map S, thử đường cong và giao lộ:

```powershell
python agent4_calibration_v27\guarded_pilot_v27.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps C,X --seeds 0 --steps 900 --out ".\agent4_v27_guarded_CX.json"
```

## Ý nghĩa kết quả

- `learned_steering_applied=true`: steering model được gửi thật vào `env.step([steering, longitudinal])`.
- `learned_longitudinal_applied=false`: ga/phanh **không** lấy từ model, mà từ speed governor.
- `model_steering_steps`: số bước thử đóng vòng (closed-loop).
- `adaptive_lookahead_steps`: số bước gần đích dùng waypoint rút ngắn; đây là **distribution shift** cần xem xét, không phải dữ liệu PDD chuẩn.
- `lane_departure_guard`: dừng thử khi độ lệch tâm làn quá ngưỡng, không khẳng định xe đã ra khỏi toàn mặt đường.
- `observation_or_model_guard`: dừng khi lấy feature/waypoint hoặc model lỗi; xem `errors`.
- `arrive_dest`: chỉ thành tích của **hybrid pilot**, không phải full Agent4.
- `step_limit`: hết bước cho phép, không có nghĩa là lỗi chương trình.

## Giới hạn quan trọng

1. Không tự động chọn gain từ V2.6 (`1.54`, `2.58`, `5.53` chỉ là chẩn đoán). Mặc định gain=1; sửa bằng `--steer-gain` chỉ sau khi đánh giá thực nghiệm. 
2. Model chưa được hiệu chuẩn ga/phanh; V2.6 không có mẫu phanh. Ở thử nghiệm này, ga/phanh lấy từ bộ điều khiển tốc độ độc lập.
3. Gần đích, phép lấy waypoint 25/45 m chuyển thành khoảng ngắn hơn nhưng mô hình chưa được học phân bố đó. Ghi rõ số bước gần đích.
4. Giới hạn lane offset=1.5 m không đảm bảo an toàn giao thông. Vẫn cần đo va chạm, lệch làn và so sánh với agent rule-based/IDM về sau.
5. Chương trình không được chạy trực tiếp trong môi trường MetaDrive của tác giả tại thời điểm phát hành. Chỉ có kiểm tra cú pháp và unit tests; cần người dùng chạy thật để xác nhận API 0.4.3.
6. Trong phiên bản 2.7, `env.step()` thực sự nhận action của mô hình; chạy trong mô phỏng, **không dùng trên xe ngoài đời**.
