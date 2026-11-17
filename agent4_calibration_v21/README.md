# Agent 4 — hiệu chuẩn V2.1 (MetaDrive 0.4.3 + PDD)

## Phạm vi

Giai đoạn này chỉ **đo phản ứng vật lý và đối chiếu tọa độ**. Không dùng model để lái xe và **không chỉnh** `shared_adapter_v2.py`, CSV huấn luyện hoặc `model_v2.joblib`.

### Chuẩn bị

Giải nén thư mục `agent4_calibration_v21` vào thư mục `dataset` đang chứa PDD, `agent4_pdd_v2` và các model V2. Không chạy lệnh trong thư mục mã nguồn MetaDrive nếu có package cùng tên gây xung đột.

```powershell
cd "D:\Học\Tri tue nhan to nang cao\lab\dataset"
python agent4_calibration_v21\test_calibration_v21.py
```

### Bước A — Đo dấu steering, ga/phanh, waypoint trong MetaDrive

```powershell
python agent4_calibration_v21\calibrate_metadrive_v21.py --out ".\metadrive_calibration_v21.json"
```

- Tạo môi trường offscreen, đường thẳng `map='S'`, `traffic_density=0`.
- Reset cùng seed trước **từng trial**, tăng tốc ổn định rồi thử `steer = 0, ±0.25, ±0.4`, và so sánh biến thiên góc/độ lệch ngang so với `steer=0`.
- Đo ga tại các mức 0, 0.2, 0.4, 0.7 từ đứng yên; phanh 0, -0.15, -0.4, -0.7 sau warmup.
- Ghi waypoint MetaDrive gốc so với waypoint 10/25/45 m dọc tâm **đoạn làn hiện tại**. Điểm đo lane chỉ để chẩn đoán, **không thay thế tuyến đường**.
- Báo cáo thử nghiệm có `early_end=true` được giữ nguyên, không đưa vào phép suy dấu chính.

Có thể tăng số seed để kiểm tra tính ổn định:

```powershell
python agent4_calibration_v21\calibrate_metadrive_v21.py --seeds 0,1,2 --out ".\metadrive_calibration_v21.json"
```

### Bước B — Đối chiếu dấu steering và waypoint của CARLA/PDD

```powershell
python agent4_calibration_v21\audit_pdd_v21.py ".\PDD" --out ".\pdd_calibration_v21.json"
```

Script đo tương quan steering PDD với thay đổi yaw 0.25 giây, kiểm tra tọa độ waypoint với `right_vector` và chuyển động sang phải sau khoảng 1 giây. Việc ghép nhãn image/step vẫn chưa được xác nhận thời gian độc lập.

### Gửi hai báo cáo

- `metadrive_calibration_v21.json`
- `pdd_calibration_v21.json`

Từ số liệu này mới quyết định hệ số đổi dấu steering, cách xác định waypoint theo tuyến đường và cách so ga/phanh. Nếu thay đổi **ý nghĩa** waypoint thì phải dựng lại dữ liệu V2 và huấn luyện lại; không được sửa mỗi adapter inference.

## Các giới hạn quan trọng

- CARLA `steer > 0` là rẽ **phải**; MetaDrive sử dụng hệ điều khiển riêng. Chỉ chấp nhận hệ số đổi dấu khi cả yaw và độ lệch ngang trong thử nghiệm MetaDrive nhất quán.
- `steer_gain`, `throttle_gain`, `brake_gain` không thể quy ra chính xác từ tương quan hành động vì mô hình xe hai simulator khác nhau. Script **không** ước lượng hoặc áp dụng các hệ số giả.
- Kịch bản đường thẳng không kiểm chứng rẽ giao lộ, tránh vật cản hoặc tỷ lệ đến đích. Đây mới là hiệu chuẩn mức thấp.
- Dữ liệu đối tượng MetaDrive hiện lấy trực tiếp từ world state đặc quyền (privileged). Không phải đầu vào camera-only.

## Đã kiểm thử

Các script đã được kiểm tra cú pháp và kiểm thử hàm toán học độc lập. Trong môi trường tạo gói không có `metadrive-simulator`; thử nghiệm vật lý phải chạy trên MetaDrive 0.4.3 ở máy của bạn.
