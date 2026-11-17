# Agent 4 V3.4 — Test khả năng tổng quát hoá (MetaDrive, mô phỏng)

## Vì sao có V3.4?

V3.3 đã đạt 10/10 lượt `arrive_dest` ở seed 28–31, tốc độ mục tiêu **3 m/s**, không có giao thông. Chưa thể khẳng định bền vững với seed mới hoặc tốc độ khác. V3.4 **không huấn luyện lại** hoặc thay đổi model V3.3; chỉ tổ chức thử nghiệm đối chứng ngoài tập huấn luyện, giữ nguyên guard và điều kiện dừng của MetaDrive.

- `baseline_3`: steering từ PDD V2 gốc (tên chứa `_3` nhưng tốc độ sẽ lấy từ `--speeds`).
- `lateral_3`: PDD V2 + residual theo quy tắc V3.1, với **cùng tốc độ**.
- `student_v33`: PDD V2 + residual học từ IDM (model V3.3), **không có IDM điều khiển**.
- Mọi cấu hình đều dùng **speed governor** theo quy tắc để ga/phanh. V3.4 không chứng minh full autonomous driving.

## Bố cục thư mục

Giải nén `agent4_robustness_v34` **ngang hàng** với `agent4_learned_residual_v33`, `agent4_controller_v30`, `agent4_controller_v31`, và các dependencies cũ trong `dataset/`.

## Bước 1 — unit tests

```powershell
python agent4_robustness_v34\test_v34.py
```

## Bước 2 — 3 m/s trên 4 seed hoàn toàn mới

```powershell
python agent4_robustness_v34\benchmark_v34.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --student ".\agent4_v33_student.joblib" --maps C,X --seeds 32-35 --speeds 3 --profiles baseline_3,lateral_3,student_v33 --steps 1400 --out ".\agent4_v34_new_seeds_3mps.json"
```

Output: `agent4_v34_new_seeds_3mps.json` + CSV tên tương tự. **24 lượt** (2 bản đồ × 4 seed × 3 cấu hình).

## Bước 3 — chỉ khi bước 2 đạt yêu cầu: thử tốc độ ngoài phân phối

```powershell
python agent4_robustness_v34\benchmark_v34.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --student ".\agent4_v33_student.joblib" --maps C,X --seeds 36-39 --speeds 2,4 --profiles lateral_3,student_v33 --steps 1400 --out ".\agent4_v34_speed_stress.json"
```

Output: 2 bản đồ × 4 seed × 2 tốc độ × 2 chế độ = **32 lượt**.

**LƯU Ý:** 2 và 4 m/s không nằm trong điều kiện huấn luyện steering của Student V3.3. Đây là kiểm tra **ngoài miền dữ liệu**, không phải tối ưu hyperparameters. Không thêm seed đánh giá V3.4 vào train/validation sau khi nhìn kết quả; nếu cần huấn luyện lại phải dùng tập dữ liệu và seed đánh giá mới.

## Đọc báo cáo

- `status`: chỉ `arrive_dest` được tính thành công. `lane_departure_guard` là dừng phòng ngừa, **không** phải `out_of_road` của MetaDrive.
- `scores_by_speed_map_profile`: tỷ lệ thành công theo tốc độ × bản đồ × mode.
- `steering_attribution_v34` trong trace của Student ghi **steering PDD nguyên gốc**, **residual học**, **steering Student đề xuất**, **steering thực sự áp dụng**. V3.3 cũ đã gộp hai thành phần đầu vào trường `model_steer_md`.
- `student_predictions_aligned`: cần `true` cho mỗi episode student; nếu `false`, kiểm tra lại mã vì việc ghép trace với dự đoán không đáng tin.
- `min_edge_clearance_m` chỉ là xấp xỉ hình học; cờ va chạm vạch đường từ MetaDrive có giá trị quyết định.
- `route_completion: null` có thể xảy ra khi guard dừng trước `env.step`, không có nghĩa completion = 0%.

## Giới hạn khoa học

- V3.4 chạy **traffic_density=0.0**, không kiểm tra tránh xe/người đi bộ.
- Student vẫn dùng thông tin hình học trực tiếp của MetaDrive, không phải camera/LiDAR thật.
- Ga/phanh vẫn từ governor độc lập, không phải một model được huấn luyện.
- Tất cả chỉ áp dụng trong môi trường mô phỏng. Không triển khai điều khiển xe thật.
