# Agent 4 — V2.5: kiểm chứng ở ĐOẠN CONG / GIAO LỘ thực (shadow)

## Vì sao không coi V2.4 là đã vượt bài thử C/X?

Báo cáo V2.4 cho map S, C, X có **ba summary y hệt**. Xe được điều khiển `steer=0`, `gas=0.28` trong 60 bước; việc đặt tên map C/X **không chứng minh** xe đã tới đường cong hoặc giao lộ. V2.5 dùng chính `IDMPolicy` (rule based) đi theo tuyến **chỉ để thu thập observation**. Model PDD V2 vẫn chỉ sinh prediction shadow và **không có action model nào đưa vào `env.step()`**.

## Giải nén

Đặt `agent4_calibration_v25/` ngang hàng với:

```
dataset/
  agent4_pdd_v2/
  agent4_calibration_v22/
  agent4_calibration_v23/
  agent4_calibration_v24/
  agent4_calibration_v25/
  pdd_v2_models/at_t/model_v2.joblib
```

Cần Python 3.10, MetaDrive 0.4.3, joblib, numpy và các thư mục mã đã chạy ở V2.4.

## Lệnh

```powershell
cd "D:\Học\Tri tue nhan to nang cao\lab\dataset"
python agent4_calibration_v25\test_coverage_v25.py
python agent4_calibration_v25\probe_coverage_v25.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps "C,X" --seeds "0,1,2" --steps 450 --out ".\agent4_v25_coverage_probe.json"
```

Có thể thử `--steps 650` nếu IDM chưa đi vào phần đường cong/giao lộ. Các episodes có thể kết thúc sớm khi đạt đích, bị crash, hoặc ra khỏi đường. Chương trình đánh dấu riêng các lỗi geometry.

## Giải thích số liệu

- `n_in_curve_lane`: số bước ego thực sự nằm trên làn có kiểu `CircularLane` (hoặc Arc); đáng tin cậy hơn chỉ nhìn tên map `C`.
- `n_in_curve_block`, `n_in_intersection_block`: số bước đường hiện tại của ego có chỉ số node nhận dạng block C/X. Đây là dấu hiệu đã đến block tương ứng, **không khẳng định tự động rẽ đúng tại giao lộ**.
- `n_turn_motion_samples`: số bước thay đổi yaw hơn `0.5°/step` (mặc định).
- `n_non_initial_road`: số bước ego không còn ở road bắt đầu.
- `n_usable_turning_samples`, `n_usable_intersection_samples`: số mẫu có thể so sánh prediction tại vùng thực sự liên quan.
- `n_lane_chain_broken`, `n_lane_horizon_incomplete`: nghi vấn không nên dùng để điều khiển.
- `episodes`: distance_travelled, max_speed, terminal condition.
- `turn_examples`, `intersection_examples`, `invalid_examples`: minh chứng chọn lọc, tránh file hàng chục MB.

**Chỉ khi coverage count > 0** mới có căn cứ nói phép thử đã quan sát đường cong/giao lộ. Cần xem sai lệch waypoint và dấu steering trong các vùng đó. Model không lái, chưa đo được tỷ lệ về đích của Agent 4.

## Giới hạn

- IDMPolicy của MetaDrive là expert tham chiếu, **không phải** Model V2 và có thể lệch luồng học PDD.
- V2.4 nối làn theo độ liên tục hình học, chưa có bảo đảm đúng làn rẽ theo lộ trình.
- Chưa hiệu chuẩn steering gain/ga/phanh CARLA→MetaDrive; không được áp dụng trực tiếp Model V2.
- Phương pháp waypoint 25m/45m vẫn khác PDD (biến thiên theo kịch bản).
- `agent_policy=IDMPolicy` đã được mô tả trong tài liệu MetaDrive, nhưng phiên bản cụ thể 0.4.3 trên máy bạn là kiểm thử cuối cùng; nếu lỗi hãy gửi traceback đầy đủ.

## Kiểm thử đã làm trong môi trường tạo code

`py_compile`, `test_coverage_v25.py` (đơn vị toán học), không có MetaDrive engine thực tế tại đây. Không huấn luyện lại model.
