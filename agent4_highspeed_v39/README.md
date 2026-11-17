# Agent 4 V3.9: Benchmark tốc độ cao (chưa huấn luyện lại)

V3.9 **không tạo model mới** và không mở giới hạn tốc độ của model cũ bằng cách giả vờ rằng nó đã học tốc độ cao. Đây là **benchmark chẩn đoán** để xác định tốc độ mục tiêu 4/6/8/10 m/s có thực sự đạt được trên MetaDrive không, và hệ thống có giữ làn/về đích hay không.

## Điều kiện cố định

- MetaDrive 0.4.3, Python tương thích môi trường dự án. `traffic_density=0`, **không tạo xe dẫn đầu**.
- Steering: PDD V2 + learned residual V3.5, **không sửa model** trong tất cả cấu hình.
- So sánh longitudinal: `governor_v35` (quy tắc), `learned_v37` (học từ governor), `learned_v38` (học tránh xe trước).
- `learned_v38` vẫn nhận các đặc trưng front_* nhưng sẽ thấy `front_present=0` vì đường trống.
- Giữ nguyên logic feature/route/steering/termination từ các phiên bản đã dùng; chỉ thay parser tốc độ mới (parser V3.4 giới hạn 5 m/s).
- Chỉ `arrive_dest` được tính thành công. Các lượt thất bại không được xếp hạng là “về đích nhanh”.
- Tất cả tốc độ >4 m/s là **ngoài tập huấn luyện** của cả steering V3.5 và longitudinal V3.7/V3.8; không sử dụng kết quả để tuyên bố model đã được đào tạo tốc độ cao.
- Để đo đúng thời gian, `physics_world_step_size=0.02`, `decision_repeat=5`, tương đương 0.1 giây mô phỏng / action; không nhầm với thời gian xử lý máy tính.

## Cài và chạy

Giải nén `agent4_highspeed_v39.zip` **ở trong thư mục dataset**, ngang hàng với `agent4_robustness_v34`, `agent4_multispeed_v35`, `agent4_onpolicy_longitudinal_v37`, `agent4_traffic_v38` và các phụ thuộc cũ. Không ghi đè mã hoặc model cũ.

### 0. Unit tests

```powershell
python agent4_highspeed_v39\test_v39.py
```

### 1. Smoke test: đường trống, 4m/s, 1 seed

```powershell
python agent4_highspeed_v39\benchmark_v39.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --longitudinal-v37 ".\agent4_v37_longitudinal.joblib" --longitudinal-v38 ".\agent4_v38_traffic.joblib" --maps S --seeds 120 --speeds 4 --profiles governor_v35,learned_v37,learned_v38 --steps 1800 --out ".\agent4_v39_smoke.json"
```

### 2. Đường trống: kiểm tra hồi quy 2,3,4 m/s

```powershell
python agent4_highspeed_v39\benchmark_v39.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --longitudinal-v37 ".\agent4_v37_longitudinal.joblib" --longitudinal-v38 ".\agent4_v38_traffic.joblib" --maps S,C,X --seeds 120,121 --speeds 2,3,4 --profiles governor_v35,learned_v37,learned_v38 --steps 1800 --out ".\agent4_v39_empty_road_regression.json"
```

54 lượt, 3 bản đồ × 2 seed × 3 mức tốc độ × 3 chính sách.

### 3. Đường thẳng dài: tốc độ 4/6/8/10 m/s

```powershell
python agent4_highspeed_v39\benchmark_v39.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --longitudinal-v37 ".\agent4_v37_longitudinal.joblib" --longitudinal-v38 ".\agent4_v38_traffic.joblib" --maps SSSSS --seeds 120,121 --speeds 4,6,8,10 --profiles governor_v35,learned_v37,learned_v38 --steps 2500 --out ".\agent4_v39_long_straight_speed_sweep.json"
```

24 lượt, 5 đoạn S nối tiếp. **Chỉ chạy sau khi smoke test PASS.** Nếu bản đồ `SSSSS` không khởi tạo được trên cấu hình MetaDrive của máy, thử `--maps SSS` hoặc `--maps S`, gửi lỗi để chẩn đoán; đừng coi chạy trên bản đồ S ngắn là đo tốc độ duy trì 10 m/s.

### 4. Nếu cần thử đường cong

```powershell
python agent4_highspeed_v39\benchmark_v39.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --longitudinal-v37 ".\agent4_v37_longitudinal.joblib" --longitudinal-v38 ".\agent4_v38_traffic.joblib" --maps C,X --seeds 120,121 --speeds 4,6,8,10 --steps 1800 --out ".\agent4_v39_curve_speed_sweep.json"
```

48 lượt. Mục đích là phát hiện ngưỡng thất bại trước khi thu thập/huấn luyện dữ liệu tốc độ cao. Không chạy trực tiếp trên xe thật.

## Cách đọc kết quả

- `.json`: toàn bộ `episodes`, `summary`, `trace`, `group_scores`. `time_to_goal_s` chỉ có nếu status `arrive_dest`.
- `.csv`: một dòng cho mỗi lượt chạy; dễ mở bằng Excel hoặc Python.
- `max_speed_mps`: tốc độ cao nhất **thực sự ghi nhận**, không phải tốc độ mục tiêu.
- `mean_speed_mps`: trung bình tốc độ các action trước khi dừng/đích.
- `time_to_goal_s`: số bước hành động × 0.1s (đã cố định trong cấu hình vật lý).
- `max_abs_planned_lateral_right_m` và `min_estimated_edge_clearance_m`: theo dõi độ ổn định quỹ đạo; giá trị clearance chỉ là ước lượng, không phải chứng nhận an toàn.
- `outside_learned_training_speeds=true` cho mốc dưới 2 hoặc trên 4m/s.
- `governor_deviation_mae`: độ lệch lệnh học so với governor tại **cùng trạng thái của chính lượt chạy đó**; không phải đánh giá offline theo tập test.

**Tiêu chí quyết định V3.10:** (1) đường dài giữ làn; (2) về đích, không va chạm; (3) đạt gần tốc độ yêu cầu; (4) thời gian về đích giảm thực sự; (5) điều khiển learned ổn định. Nếu governor thành công nhưng model thất bại ở >4m/s, ưu tiên **thu thập dữ liệu tốc độ cao và huấn luyện lại steering/longitudinal**, không ép model cũ extrapolate.

**Quy ước seed:** `120,121` chỉ dành cho benchmark thăm dò V3.9; không đào tạo trên kết quả của chúng rồi gọi đó là final holdout. Khi phát triển V3.10, chọn nhóm seed train/val mới không trùng và để dành seed độc lập cuối cùng.
