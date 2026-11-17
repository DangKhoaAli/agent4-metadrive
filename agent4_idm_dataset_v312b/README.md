# Agent 4 V3.12B — Verified IDM Full-Step Data Collector

**Mục tiêu:** Tạo dataset **imitation learning** từ MetaDrive IDMPolicy trên `S`, `C`, `X`, lưu **mỗi bước** (không phải trace thưa như V3.12), chấm độ tin cậy nhãn steering và tách tập theo **seed**. Không huấn luyện bất kỳ model nào ở giai đoạn này.

## 0. Chuẩn bị

Giải nén ZIP vào cùng thư mục `dataset` mà các bản V3.12 trước đã chạy:

```text
dataset/
  agent4_idm_dataset_v312b/
    collect_v312b.py
    geometry_v312.py
    validate_v312b.py
    test_v312b.py
    dataset_schema.json
    README.md
```

Yêu cầu **MetaDrive 0.4.3** trong môi trường Python hiện có. Không cần tải `.joblib` PDD hoặc V3.10 để thu thập IDM. **Không chạy lại file JSON V3.12** để tạo training rows: V3.12 chỉ lưu trace lấy mẫu mỗi 15 bước; bản mới phải mô phỏng để ghi đủ từng bước.

## 1. Kiểm thử offline (không cần MetaDrive)

```powershell
python agent4_idm_dataset_v312b\test_v312b.py
```

8 unit tests gồm kiểm tra nhãn raw/applied, phát hiện hành động vượt giới hạn, nối lookahead qua đầu làn, kiểm tra seed không trùng, cách ly episode thất bại và resume. Unit tests trên **mô phỏng giả**, không phải bằng chứng chương trình chạy MetaDrive thật.

## 2. Smoke test — chỉ 6 episode

**Phải chạy trước batch lớn.** Đặt trong thư mục `dataset`:

```powershell
python agent4_idm_dataset_v312b\collect_v312b.py --maps C,X --speeds 3 --train-seeds 170 --val-seeds 171 --test-seeds 172 --steps 2500 --out-dir ".\agent4_v312b_smoke"
```

Sau đó kiểm tra dữ liệu bằng script độc lập, cũng không cần khởi động MetaDrive:

```powershell
python agent4_idm_dataset_v312b\validate_v312b.py --dataset ".\agent4_v312b_smoke"
```

Mở `agent4_v312b_smoke/manifest.json`, kiểm tra `totals`, các episode `status`, `quality_pass`, `raw_out_of_range_steps`, và `errors`. **Nếu có `episode_error` do `current_action` không khớp hoặc lỗi API, ngừng chạy batch lớn và gửi manifest + audit của episode lỗi. Không gỡ bước kiểm chứng nhãn.**

## 3. Batch chính — 216 episode

Chỉ chạy sau khi smoke test hợp lệ:

```powershell
python agent4_idm_dataset_v312b\collect_v312b.py --maps S,C,X --speeds 2,3,4 --train-seeds 170-185 --val-seeds 186-189 --test-seeds 190-193 --steps 2500 --out-dir ".\agent4_v312b_idm_dataset"
```

Phân chia **16 train + 4 val + 4 test** seed, 3 bản đồ × 3 tốc độ = **216 episode**. Đây là kế hoạch số lượt, **không phải số episode đã chạy**. Tốc độ `--speeds` có đơn vị **m/s**; bộ thu đặt `IDMPolicy.NORMAL_SPEED = speed × 3.6 km/h`, và luôn ghi **tốc độ thực tế**.

Nếu batch bị ngắt, chạy lại **cùng lệnh** với `--resume`, ví dụ:

```powershell
python agent4_idm_dataset_v312b\collect_v312b.py --maps S,C,X --speeds 2,3,4 --train-seeds 170-185 --val-seeds 186-189 --test-seeds 190-193 --steps 2500 --out-dir ".\agent4_v312b_idm_dataset" --resume
```

Sau khi chạy xong:

```powershell
python agent4_idm_dataset_v312b\validate_v312b.py --dataset ".\agent4_v312b_idm_dataset"
```

Không dùng `--resume` với thông số khác, vì bộ thu sẽ **từ chối nối vào dataset sai cấu hình**.

## 4. Cấu trúc kết quả

```text
agent4_v312b_idm_dataset/
  manifest.json
  audit/
    train/C/seed170_speed3p0mps.csv
    val/C/...
    test/X/...
  clean/
    train/C/seed170_speed3p0mps.csv
    val/C/...
    test/X/...
```

- **audit/**: tất cả bước đã có nhãn được kiểm chứng, kể cả episode thất bại; có cả `raw_steering` từ IDM và `applied_steering` trên xe.
- **clean/**: chỉ episode `arrive_dest` + qua tiêu chí chất lượng; loại hàng có `raw_steering` hoặc `raw_throttle_brake` ngoài `[-1,1]`, hoặc thiếu lookahead 5/10 m, hoặc không đọc được clearance proxy. Không tự gán nhãn từ hành động thô vượt chuẩn.
- **manifest.json**: mọi episode, số bước, lỗi, số nhãn bất thường, chất lượng và tổng hợp `train/val/test`.

**Nhãn chính:** `applied_steering`, `applied_throttle_brake`, đọc từ `vehicle.current_action` **sau env.step**, và so khớp `vehicle.steering`, `vehicle.throttle_brake`, `clip(IDM.act())` trong cùng bước. Nếu không khớp, đánh dấu episode lỗi, tuyệt đối không đưa dữ liệu vào `clean/`.

**Đầu vào gợi ý**: `speed_mps`, `planned_lateral_right_m`, `heading_error_rad`, `distance_to_lane_end_m`, `curvature_proxy_rad_per_m`, `lookahead_heading_5m_rad`, `lookahead_heading_10m_rad`, `lookahead_heading_20m_rad`, `is_circular_lane`. Xem `dataset_schema.json`.

**Lookahead:** tìm làn nối tiếp có vị trí đầu gần vị trí cuối của làn hiện tại (sai khác tối đa 2 m). Không mặc định `next_ref_lanes[0]`, vì trên đường nhiều làn, nó có thể trỏ nhầm làn. Nếu không thể xác minh, đánh dấu `lookahead_Nm_available=0`.

**Tiêu chí chấp nhận episode mặc định:** đến đích, không lỗi; độ lệch ngang cực đại ≤0.60 m; heading error ≤0.12 rad; khoảng cách mép đường ước lượng ≥0.30 m; không có bước cảnh báo clearance `<0.35 m`; không có bước `on_yellow_continuous_line`. Có thể thay đổi ngưỡng qua CLI nhưng phải **tạo dataset mới**, không nối với dataset cũ. Các tiêu chí mang tính heuristic cho mô phỏng, không chứng nhận an toàn.

**Chia tập:** Tách theo seed để tránh cùng một hành trình xuất hiện trong train và test. Dữ liệu từ bản đồ cùng loại vẫn có thể có các hình học tương đồng; chưa thể gọi là generalization sang bản đồ mới nếu chỉ dùng seed hold-out.

## 5. Điều kiện chuyển sang V3.13 (huấn luyện)

1. Smoke test thực tế không xảy ra mismatch API/action; `validate_v312b.py` báo PASS.
2. `manifest.json` có đủ 3 split và số `clean_steps` đủ lớn trong mỗi split.
3. Đánh giá tỷ lệ episode clean trên từng map/speed, tình huống X có/không có `CircularLane`, và phân phối curvature/steering.
4. Huấn luyện thử mô hình steering trên **train**, chọn hyperparameter trên **val**, chốt trên **test**. Không dùng thông tin `post_*`, `seed`, `split`, hoặc hành động IDM thô làm feature đầu vào.
5. Đánh giá **closed-loop** trên MetaDrive trên seed/test chưa dùng huấn luyện, đặc biệt `C/X`, và so sánh V3.10 lẫn IDM. Không được dùng IDM fallback trong chế độ learned khi báo cáo Agent 4.

**Giới hạn:** Bản ZIP có kiểm thử offline; môi trường tạo ZIP không cài MetaDrive nên **chưa thể xác nhận tương thích runtime 0.4.3 tại đây**. Cơ chế nhãn fail-closed được thêm để phát hiện sai khác API khi chạy máy thật của bạn.

Nguồn API tham khảo: [MetaDrive Action and Policy](https://metadrive-simulator.readthedocs.io/en/latest/action.html) và `metadrive/component/vehicle/base_vehicle.py` (trường `current_action`, `_preprocess_action`, `steering`, `throttle_brake`).
