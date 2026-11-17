# Agent 4 — V3.16.1: ổn định đầu ra AI ga/phanh (thử nghiệm có kiểm soát)

**Trạng thái:** đã chạy **37 unit tests offline**, **chưa chạy MetaDrive thực tế**. V3.16.1 **không huấn luyện model mới**: giữ nguyên `agent4_v316_longitudinal.joblib`, thêm một bộ xử lý tín hiệu **có trạng thái** trên đầu ra của model để kiểm tra giả thuyết lệnh ga/phanh quá gắt/đảo chiều. Chỉ chọn nếu closed-loop validation chứng minh không suy giảm. Không sửa mã/model V3.12B, V3.13, V3.15, V3.16.

## Chẩn đoán đã xác nhận từ `agent4_v316_validation.json`

- 36 kịch bản, 72 lượt: rule governor 35/36 quality, V3.16 34/36 quality; cả hai 36/36 về đích.
- V3.16 MAE tốc độ 0.17283 m/s so với 0.21220 m/s của rule, nhưng phát lệnh phanh 4,292 bước so với 454 bước của rule. **Số bước phanh lớn không tự chứng minh sự dao động**, nhưng là điểm đáng kiểm tra.
- V3.16 giảm chất lượng trên `C/186/2` (26 cảnh báo mép đường) và `C/189/2` (độ lệch ngang tối đa khoảng 61.5 cm); cải thiện `C/186/3`.
- Lưu ý: log `trace` gốc chỉ lấy mẫu mỗi 30 bước, nên **không thể xác nhận toàn bộ tần suất dao động lệnh từ log cũ**. V3.16.1 bổ sung chỉ số `action_total_variation` và `action_sign_reversals` cho các lượt mới.

## Thuật toán V3.16.1

`raw = predict_long(v316_bundle, pre_step_state, features_v313)`.

Bộ lọc áp dụng lần lượt: clamp biên độ (+0.70/−0.80), deadband 0.065, EMA bất đối xứng (nhả/tăng ga 0.30; giảm ga/phanh 0.65), hạn chế thay đổi lệnh mỗi bước (+0.16/−0.35). **Đây chỉ là mặc định thử nghiệm, không phải tham số đã được chứng minh tốt nhất.** Bộ lọc reset mỗi episode. Không gọi IDM, không gọi bộ governor ở nhánh V3.16/V3.16.1; bộ governor chỉ dùng trong nhánh so sánh baseline.

Việc làm mượt có thể **làm chậm phanh** trong các pha chuyển tốc độ đột ngột. Vì thế phải kiểm tra cả `overspeed_steps`, `quality_pass`, `speed_mae_mps`; không dùng chỉ số giảm số bước phanh để kết luận an toàn.

## Bố trí thư mục

Giải nén ZIP bên cạnh thư mục V3.16:

```text
<dataset>/
  agent4_longitudinal_v316/          # đã có, KHÔNG sửa
  agent4_longitudinal_v3161/         # thư mục mới
  agent4_recovery_v315/             # đã có
  agent4_steering_v313/              # đã có
  agent4_idm_dataset_v312b/         # đã có
  agent4_v312b_idm_dataset/         # đã có
  agent4_v313_model/
  agent4_v316_model/
  agent4_v316_validation.json        # log 72 lượt đã chạy
```

Cùng môi trường Windows/Python đã chạy thành công V3.16. Yêu cầu `joblib`, `numpy`, `scikit-learn`, `metadrive` từ môi trường hiện tại.

## Bước 1 — Unit test (không chạy MetaDrive)

```powershell
python .\agent4_longitudinal_v3161\test_v3161.py
```

## Bước 2 — Kiểm tra chẩn đoán V3.16 cũ (tùy chọn)

```powershell
python .\agent4_longitudinal_v3161\analyze_v316.py --input .\agent4_v316_validation.json
```

## Bước 3 — Kiểm thử nhỏ đúng hai kịch bản đã trượt

Chạy từ thư mục `<dataset>`:

```powershell
python .\agent4_longitudinal_v3161\eval_v3161.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --model .\agent4_v316_model\agent4_v316_longitudinal.joblib --reference .\agent4_v316_validation.json --maps C --seeds 186,189 --speeds 2 --out .\agent4_v3161_probe_C2.json
```

Đây là **2 lượt MetaDrive mới** (V3.16.1), cộng với **4 lượt tham chiếu nguyên bản** (2×rule, 2×V3.16) từ `agent4_v316_validation.json`. Chúng **không phải các lượt được chạy lại**. Kiểm tra `summary`, `decision.regressions`, `brake_steps` và chi tiết các episode. Trường `decision.accepted` **luôn false đối với phép thử 2 kịch bản**: chỉ validation đủ 36 kịch bản mới có quyền chấp nhận.

Nếu mất điện/dừng giữa chừng, chạy lại chính lệnh trên **thêm `--resume`**. Không đổi model/tham số khi resume.

**Gửi `agent4_v3161_probe_C2.json` trước khi chạy validation 36 kịch bản.** Nếu bộ lọc làm mất lượt về đích hay chất lượng, không chạy full; sẽ thiết kế phương pháp thu thập lại dữ liệu on-policy thay vì cố tuning các seed đã biết.

## Bước 4 — Full validation (CHỈ nếu probe ổn)

```powershell
python .\agent4_longitudinal_v3161\eval_v3161.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --model .\agent4_v316_model\agent4_v316_longitudinal.joblib --reference .\agent4_v316_validation.json --maps S,C,X --seeds 186-189 --speeds 2,3,4 --out .\agent4_v3161_validation.json
```

Bộ đánh giá reuse **72 lượt cũ** được kiểm tra SHA/metadata, chạy **36 lượt V3.16.1 mới**. Không đọc lại test `190–193`, holdout V3.15 `194–201` hay seed `>=202` để điều chỉnh. `decision.accepted` chỉ true khi đủ đúng 36 kịch bản validation và **không giảm chất lượng/về đích so với cả rule và V3.16**, MAE không vượt ngưỡng, có phanh AI, tổng số bước phanh AI ít hơn V3.16.

Nếu validation thành công mới cân nhắc final độc lập với các seed >=202. Không chạy final khi chưa được chấp nhận.

## Ý nghĩa kết quả

- **Thành công:** V3.16.1 đạt chất lượng, ổn định hơn mà không cần huấn luyện lại; đóng băng config trước final.
- **Thất bại:** không tự đổi nguồn học hay dùng governor ở nhánh AI; tiến tới bổ sung **on-policy DAgger longitudinal** tập trung trạng thái tốc độ thấp trên bản đồ C. Những kịch bản validation 186–189 chỉ được dùng để so sánh/chọn, không được trộn vào train.
- Đánh giá 0 traffic trên 3 map và privileged state; **không phải bằng chứng xe an toàn trong giao thông thực tế**.

## Các file

- `common_v3161.py`: lớp `StabilizedLongitudinal` và config.
- `eval_v3161.py`: A/B/C validation hoặc final độc lập, `--reference` reuse log validation đã xác minh.
- `analyze_v316.py`: chẩn đoán regressions của report V3.16 cũ.
- `test_v3161.py`: unit tests offline.
