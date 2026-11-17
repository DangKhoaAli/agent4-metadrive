# Agent 4 — PDD V2 & MetaDrive 0.4.3 bridge (PROTOTYPE)

Dùng với dự án MetaDrive và các tập CSV đã sinh bởi `prepare_pdd_pilot.py`.

## 1. Mục đích

- `shared_adapter_v2.py`: định nghĩa **15 đặc trưng chung** và chuyển đổi từ PDD/MetaDrive.
- `prepare_pdd_v2.py`: đọc `PDD/driver_*/data/...` + bộ `pdd_pilot_output`, xuất các CSV V2 **giữ nguyên split theo kịch bản** và nhãn cũ.
- `probe_metadrive_v2.py`: lấy trạng thái thật của MetaDrive, kiểm tra shape/đơn vị và so sánh phân bố PDD–MetaDrive.
- `train_pdd_v2.py`: huấn luyện 2 mô hình dự đoán riêng lệnh lái và ga/phanh, kiểm tra nhiều chỉ số kể cả rẽ/phanh.
- `dryrun_agent4_v2.py`: chạy pipeline MetaDrive → adapter → model **NHƯNG KHÔNG ÁP DỤNG ACTION CỦA MODEL** (chưa được hiệu chuẩn).
- `test_shared_adapter_v2.py`: tự kiểm tra thuật toán tính đặc trưng.

## 2. Cách chạy trên Windows PowerShell

Giải nén `agent4_pdd_v2.zip` vào `D:\Học\Tri tue nhan to nang cao\lab\dataset`, sao cho có thư mục `dataset\agent4_pdd_v2\`.

```powershell
cd "D:\Học\Tri tue nhan to nang cao\lab\dataset"
python agent4_pdd_v2\test_shared_adapter_v2.py
python agent4_pdd_v2\prepare_pdd_v2.py ".\PDD" --pilot ".\pdd_pilot_output" --out ".\pdd_v2_output"
python agent4_pdd_v2\probe_metadrive_v2.py --pdd-v2 ".\pdd_v2_output" --out ".\metadrive_v2_probe.json"
```

Trước khi huấn luyện: xem `pdd_v2_output/metadata_v2.json` và `metadrive_v2_probe.json`.

Nếu **không có errors** trong probe, các giá trị không bị lệch đơn vị/hệ trục nghiêm trọng và tỉ lệ mẫu bị loại thấp, có thể **thử huấn luyện offline**:

```powershell
python -m pip install numpy pandas scikit-learn joblib
python agent4_pdd_v2\train_pdd_v2.py ".\pdd_v2_output" --label at_t --out ".\pdd_v2_models"
python agent4_pdd_v2\dryrun_agent4_v2.py ".\pdd_v2_models\at_t\model_v2.joblib" --out ".\agent4_dryrun_v2.json"
```

Các file báo cáo quan trọng:
- `pdd_v2_output/metadata_v2.json` — feature/schema, số bản ghi, phân bố PDD, tỉ lệ mất dữ liệu.
- `metadrive_v2_probe.json` — kiểm tra observation MetaDrive và chênh lệch phân bố.
- `pdd_v2_models/at_t/metrics_v2.json` — so sánh lỗi model theo lái/phanh với baseline.
- `agent4_dryrun_v2.json` — inference thật từ MetaDrive, **không lái xe bằng model**.

## 3. Bộ 15 đặc trưng, đúng thứ tự

```
speed_mps
wp1_forward_m, wp1_right_m
wp2_forward_m, wp2_right_m
lane_width_m
front_vehicle_dist_m, front_vehicle_closing_speed_mps
left_vehicle_dist_m, right_vehicle_dist_m
front_pedestrian_dist_m
has_front_vehicle, has_left_vehicle, has_right_vehicle, has_front_pedestrian
```

Quy ước: +x phía trước xe, +y bên phải; waypoint được giới hạn trong bán kính 50m (theo MetaDrive navigation 0.4.3); khoảng cách vật cản tối đa 60m + mask. Khoảng cách tính đến **tâm đối tượng**, không phải mép xe. `closing_speed` chỉ là chênh lệch độ lớn tốc độ (KHÔNG phải vận tốc tương đối có hướng).

PDD có thể khác dấu trục y so với MetaDrive. Để thử phiên bản ngược dấu khi có bằng chứng dữ liệu thực tế, dùng:

```powershell
python agent4_pdd_v2\prepare_pdd_v2.py ".\PDD" --pilot ".\pdd_pilot_output" --out ".\pdd_v2_output_reversed" --box-y-sign -1 --waypoint-y-sign -1
```

**KHÔNG chọn dấu theo điểm MAE offline đơn lẻ**; phải kiểm tra vật cản bên trái/phải và hướng quay của xe.

## 4. Cảnh báo chuyên môn / hạn chế

1. PDD lưu ảnh ở 4 FPS, có 5 bước control giữa hai khung telemetry. Ghép `frame i -> step 5*i` dựa trên bằng chứng thống kê, chưa được chứng nhận bằng mã thu thập gốc.
2. Waypoint CARLA và MetaDrive khác **vị trí/thuật toán checkpoint**. Adapter đồng nhất đơn vị/hệ trục và cắt ở 50m, nhưng **chưa thể bảo đảm cùng ý nghĩa điều hướng**.
3. Thông tin xung quanh của MetaDrive được lấy từ trạng thái đối tượng mô phỏng (privileged information), **không phải từ ảnh camera RGB**.
4. Bộ này chưa mô tả đầy đủ vật cản tĩnh, tín hiệu đèn giao thông, giao lộ ưu tiên, ý định chuyển làn. Những tình huống này có thể làm Agent 4 thất bại.
5. Dự đoán `steer` và `throttle-brake` từ CARLA **chưa được hiệu chuẩn động lực học trên MetaDrive**. Các script cố ý không cho model điều khiển xe thật.
6. Tập `test.csv` chỉ dùng đánh giá cuối cùng, không chọn model.
7. PDD phát hành theo CC BY-NC 4.0; kiểm tra giấy phép khi sử dụng ngoài nghiên cứu phi thương mại.

## 5. Tiêu chí có thể đi tiếp

- PDD V2 có đủ dữ liệu; hàng bị loại dưới 1% (script báo lỗi nếu trên mức này).
- MetaDrive adapter xuất đúng 15 feature, không NaN/Inf; báo cáo không có errors.
- Hai bên có phân bố/đơn vị hợp lý và đủ trường hợp có xe, người đi bộ; kiểm tra tính đúng của trái/phải.
- Model mới cải thiện so với constant trên validation **và** có lỗi rẽ/phanh chấp nhận được.
- Sau đó **mới** triển khai Agent 4 và kiểm tra closed-loop trên nhiều map/seed; phải kiểm chứng dấu steer, hiệu quả ga/phanh, và cách chọn waypoint.

## 6. Mức độ đã kiểm thử trong gói này

- Đã chạy `py_compile`, `test_shared_adapter_v2.py` và vòng xử lý/huấn luyện trên **dữ liệu mẫu mô phỏng**.
- **Chưa chạy MetaDrive 0.4.3 trên máy bạn** và chưa truy cập bản PDD gốc 7.875 mẫu từ máy bạn trong phiên tạo gói. Hãy dùng kết quả `probe_metadrive_v2.py` để xác thực tương thích bản cài đặt.
