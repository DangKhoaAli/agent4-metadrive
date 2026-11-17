# Agent 4 Calibration V2.3 — Route-aware waypoint shadow probe

## Mục tiêu

**V2.2** đã hiệu chỉnh dấu trục ngang và xác định hệ số chiều steering `-1` (CARLA sang MetaDrive). **V2.3** kiểm tra một phương án mới cho waypoint: chọn hai điểm **25 m / 45 m dọc tuyến đường quy hoạch**, thay vì lấy nguyên checkpoint của MetaDrive (có thể chỉ cách xe vài mét).

Phương án được xây dựng dựa trên MetaDrive 0.4.3 `NodeNetworkNavigation`: `checkpoints`, `_target_checkpoints_index`, `map.road_network.graph`, `lane.position(longitudinal, lateral)` và cùng công thức lấy tâm tuyến đường trong navigation. Nó phụ thuộc API nội bộ và dùng **privileged route state**, không phải mô hình camera-only.

**Không sửa model PDD V2, không đào tạo lại, không áp model điều khiển xe.** Quan trọng: target_point của PDD có độ xa thay đổi, vì vậy waypoint 25/45 m cố định của V2.3 **chưa tương đương chính thức** với PDD. V2.3 là phép thử so sánh, không phải bản sửa cuối cùng.

## Cấu trúc thư mục

```
dataset/
  agent4_pdd_v2/
  agent4_calibration_v22/
  agent4_calibration_v23/
  pdd_v2_models/at_t/model_v2.joblib
```

## Chạy (Windows)

```powershell
cd "D:\Học\Tri tue nhan to nang cao\lab\dataset"
python agent4_calibration_v23\test_route_lookahead_v23.py
python agent4_calibration_v23\probe_route_v23.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --out ".\agent4_v23_route_probe.json"
```

Chương trình lái tham chiếu bằng `[0.0, 0.28]` trên bản đồ thẳng `S`, 4 seed × tối đa 80 bước. **Không gửi action dự đoán vào `env.step()`**. File JSON tổng hợp độ xa hai waypoint trước/sau, các dự đoán steering trước/sau, cờ `complete`, `geometry_continuous` và lỗi từng seed.

Sau khi bước trên chạy ổn trên bản đồ `S`, có thể thử thêm một map cong nếu MetaDrive chấp nhận map string đó (ví dụ `--map C`) nhưng cần đánh giá kết quả thực tế thay vì mặc định map string hợp lệ.

## Điều kiện để đi tiếp

- `errors: []`, `summary.n_usable > 0`.
- Kiểm tra `n_incomplete_route` và `n_geometry_discontinuous` trước khi dùng.
- Quan sát steering khi waypoint thay đổi; **không chỉ so điểm trung vị**.
- Hiệu chỉnh dynamics CARLA/MetaDrive và làm closed-loop thử nghiệm có giới hạn mới có thể đánh giá Agent 4.

Nếu `n_incomplete_route` tăng ở cuối chuyến thì do đường còn lại không đủ 45 m, không nhất thiết là bug; cần policy xử lý gần đích.
