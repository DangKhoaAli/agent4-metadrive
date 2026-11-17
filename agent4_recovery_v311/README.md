# Agent 4 V3.11-A — Chẩn đoán khả năng phục hồi lệch làn

## Mục tiêu

Đo **khả năng phục hồi** của steering V3.10 khi có ngoại lực gây sai lệch (steering kick), thay vì chỉ đánh giá độ lệch khi xe chạy bình thường. So sánh độc lập ba bộ điều khiển trên **cùng seed / tốc độ / bản đồ / tín hiệu nhiễu**:

- `pdd_only`: PDD V2 học từ dữ liệu CARLA, không hiệu chỉnh.
- `rule_lateral`: PDD V2 + hiệu chỉnh lái giáo viên bằng quy tắc.
- `learned_v310`: PDD V2 + residual steering do model V3.10 dự đoán; **không fallback** sang giáo viên.

Tất cả cùng sử dụng **governor ga/phanh quy tắc**. Đây không phải kiểm chứng toàn bộ Agent 4 học máy, không dùng RL, không có giao thông và không có đường cong.

## Cấu trúc thư mục

Giải nén `agent4_recovery_v311/` trực tiếp vào thư mục `dataset/`, cùng cấp với `agent4_highspeed_training_v310/`, `agent4_controller_v30/`, `agent4_controller_v31/` và các phiên bản phụ thuộc trước đó. Cần Python 3.10, MetaDrive 0.4.3, `numpy`, `scikit-learn`, `joblib` và file model V3.10.

## Hướng dẫn PowerShell

**1. Test offline (không dùng MetaDrive):**

```powershell
python agent4_recovery_v311\test_v311.py
```

**2. Smoke test ngắn trên MetaDrive (3 lượt):**

```powershell
python agent4_recovery_v311\recovery_v311.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v310_steering.joblib" --maps S --seeds 150 --speeds 4 --profiles pdd_only,rule_lateral,learned_v310 --targets 0.2 --directions positive --steps 900 --out ".\agent4_v311_smoke.json"
```

**3. Benchmark chính thức — thử lệch mục tiêu 20 cm trên đường thẳng dài (48 lượt):**

```powershell
python agent4_recovery_v311\recovery_v311.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v310_steering.joblib" --maps SSSSS --seeds 150,151 --speeds 4,6,8,10 --profiles pdd_only,rule_lateral,learned_v310 --targets 0.2 --directions positive,negative --steps 2500 --out ".\agent4_v311_recovery_20cm.json"
```

**4. Chỉ khi 20 cm có kết quả chấp nhận được, thử 40 cm (48 lượt):**

```powershell
python agent4_recovery_v311\recovery_v311.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v310_steering.joblib" --maps SSSSS --seeds 152,153 --speeds 4,6,8,10 --profiles pdd_only,rule_lateral,learned_v310 --targets 0.4 --directions positive,negative --steps 2500 --out ".\agent4_v311_recovery_40cm.json"
```

Các seed 150–153 là seed **chưa dùng trong huấn luyện/validation V3.10** (train 130–141, val 142–145). Chú ý: một seed đã được dùng để chỉnh sửa siêu tham số thì không còn là *final holdout* sau đó.

## Cơ chế thử và giới hạn

Tại step 65, công cụ bơm thêm steering cố định ±0.10 vào lệnh do model/teacher/PDD dự đoán. Xung được ngắt khi:

- độ lệch ngang thực tế đạt ngưỡng mục tiêu (`--targets 0.2` hoặc `0.4`), hoặc
- đạt 22 bước xung, hoặc
- đạt ngưỡng bảo vệ 0.85 m / 0.20 rad.

Sau xung, controller được **trả toàn quyền đánh lái**. Không dùng quy tắc hỗ trợ phục hồi cho `learned_v310`. Đo **recovery** khi độ lệch <= 0.05 m và sai số hướng <= 0.02 rad liên tục **10 bước**, trong vòng **100 bước (10 giây)** sau khi ngắt xung.

**Quan trọng:** Đây là xung lái, không phải teleport xe lệch chính xác 20/40 cm. `target_reached=false` nghĩa là xung không tạo được tình huống lệch đủ lớn; trường hợp đó **không thể tính là phục hồi thành công**. Bên `positive` và `negative` chỉ là hai chiều lệnh đánh lái, không suy diễn thành trái/phải mặt đường.

## Đọc kết quả

Xuất `*.json` và `*.csv` (cùng tên). Trong mỗi episode:

- `summary.status`: `arrive_dest`, `out_of_road`, `lane_departure_guard`, ...
- `disturbance.target_reached`: nhiễu thực sự đạt mục tiêu độ lệch hay chưa.
- `disturbance.peak_abs_offset_m`: độ lệch cực đại thực tế, **không phải** độ lệch yêu cầu.
- `disturbance.recovered`, `recovery_outcome`: có phục hồi và giữ ổn định liên tục không.
- `disturbance.recovery_time_s`: thời gian từ ngắt xung đến đầu khoảng 10 bước ổn định.
- `summary.student_predictions_aligned`: chỉ số bắt buộc cho `learned_v310`.
- `summary.measured_steps_aligned`: số bước quan sát phải khớp số bước điều khiển.
- `aggregate`: thống kê `targets_reached`, `recovered` và tỷ lệ phục hồi **chỉ tính trong các lượt đạt mục tiêu**.

Ưu tiên **tỷ lệ về đích + tỷ lệ phục hồi**, không chỉ MAE hoặc độ lệch trung bình. Các lượt `target_not_reached`, `not_triggered` và `insufficient_observation` không được coi là thành công, không được âm thầm loại khỏi báo cáo. Nếu giáo viên thất bại ở mức nhiễu nào đó, chưa vội huấn luyện lại học sinh từ giáo viên đó.

## Trạng thái kiểm chứng

`test_v311.py` kiểm tra hợp đồng dữ liệu và một MetaDrive giả lập tối giản, **không kiểm thử vật lý MetaDrive thực tế**. Cần chạy smoke trên máy người dùng trước khi mở rộng. Không dùng trong xe thật.

## Sửa lỗi đo tốc độ và kiểm chứng sau khi đạt tốc độ mục tiêu

Bản vá sửa `pre_kick_state.speed_mps` / `trace.pre_speed_mps` dùng `agent.speed_km_h / 3.6` giống feature adapter V2, thay vì chia `agent.speed` thêm một lần. Không thay đổi công thức steering, governor, cơ chế bơm nhiễu hoặc bộ chấm phục hồi. Kết quả cũ về về đích / phục hồi vẫn có giá trị; tốc độ trong trace cũ thấp hơn thực tế khoảng 3.6 lần.

Để kiểm chứng nhiễu **sau khi xe tăng tốc**, sử dụng `--kick-step 150` và kiểm tra `pre_kick_state.speed_mps` trong JSON; kết quả này mới chứng minh khả năng phục hồi tại tốc độ gần mục tiêu.

```powershell
python agent4_recovery_v311\recovery_v311.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v310_steering.joblib" --maps SSSSS --seeds 152,153 --speeds 4,6,8,10 --profiles pdd_only,rule_lateral,learned_v310 --targets 0.2 --directions positive,negative --kick-step 150 --steps 2500 --out ".\agent4_v311_recovery_late_kick.json"
```
