# Agent 4 V3.16 — Học điều khiển ga/phanh từ IDM (standalone)

**Trạng thái:** Mã nguồn mới, chưa chạy MetaDrive trên máy người dùng. Không sửa V3.12B/V3.13/V3.14/V3.15.

## Thiết kế

- **Steering luôn là V3.13 được đóng băng** trong cả hai nhánh đánh giá, vì V3.15 còn một ca giảm chất lượng trên holdout.
- V3.16 học đầu ra **throttle/brake có dấu** của IDM. Dương = ga, âm = phanh; mô hình `HistGradientBoostingRegressor`, không phải heuristic lúc suy luận.
- `collect_v316.py` chạy **V3.13 + rule governor** để sinh trạng thái. Một chính sách **IDM bóng (shadow)** chỉ đưa nhãn `teacher_longitudinal`, không được gắn vào engine và không điều khiển `env.step`. Lệnh của learner được đối soát với `vehicle.current_action` sau bước. Nhãn IDM này **không phải hành động thực thi** trên xe ở lượt thu thập; tuyệt đối không mô tả ngược.
- Mục tiêu tốc độ thay đổi có chủ ý: 65 bước bình thường, 55 bước chậm đến 55%, 60 bước bình thường, lặp chu kỳ. Việc chọn `target_speed_mps` vẫn là **quy tắc/planner**, chỉ có lệnh ga/phanh được học.
- Huấn luyện **chỉ trên train seeds** của V3.12B (mặc định 170–177) + các mẫu IDM gốc `clean/train/` đã được chứng thực. Validation dùng **chỉ val seeds** 186–189. Không đọc test 190–193 hoặc final holdout V3.15 194–201 để huấn luyện/chọn model.
- Bắt buộc có **ít nhất 30 mẫu phanh** trong train và **10 mẫu phanh** trong val, nếu không thì báo lỗi để kiểm tra, không được tự nhận AI biết phanh.
- Validation A/B trên cùng seed, map, tốc độ: `rule_governor_v313` so với `learned_longitudinal_v316`; không IDM fallback, không governor fallback ở nhánh học. Chỉ đề xuất nâng cấp nếu không có ca về đích/chất lượng giảm, MAE tốc độ không quá baseline + max(0.05 m/s, 5%), và model phát lệnh phanh ít nhất 10 bước. Đây là các tiêu chí thử nghiệm, không chứng nhận an toàn ngoài đời.
- Sau khi chốt, final holdout phải dùng seed **mới từ 202 trở đi**. Seed 194–201 đã được dùng đánh giá V3.15, không còn là holdout độc lập cho V3.16.

## 1. Bố trí thư mục trên Windows

Giải nén `agent4_longitudinal_v316/` **vào cùng thư mục `dataset/`** có các thư mục đã chạy:

```text
dataset/
  agent4_idm_dataset_v312b/      (mã gốc)
  agent4_steering_v313/          (mã gốc)
  agent4_recovery_v315/          (mã gốc)
  agent4_v312b_idm_dataset/      (dataset, có manifest.json và clean/train/*.csv)
  agent4_v313_model/             (có agent4_v313_steering.joblib)
  agent4_longitudinal_v316/      (mã V3.16 mới; chỉ thêm file)
```

Yêu cầu: cùng môi trường Python với MetaDrive 0.4.3, `numpy`, `scikit-learn`, `joblib`.

## 2. Kiểm thử offline trước

Tại PowerShell ở `dataset/`:

```powershell
python .\agent4_longitudinal_v316\test_v316.py
```

Các test chỉ chứng thực logic offline, **không** xác nhận runtime MetaDrive.

## 3. Smoke test: 2 episode, rồi đọc manifest

```powershell
python .\agent4_longitudinal_v316\collect_v316.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --maps C --train-seeds 170 --val-seeds 186 --speeds 3 --steps 800 --out-dir .\agent4_v316_smoke
```

Kiểm tra `agent4_v316_smoke/collection_manifest.json`: các episode không có `errors`, `steps == verified_learner_steps`, `valid_labels > 0`, `teacher_brake_labels > 0` ở ít nhất một lượt. Nếu `shadow_error` hoặc không có phanh, **không chạy batch lớn**; gửi manifest để sửa tương thích với môi trường MetaDrive.

## 4. Thu thập chính (108 episode)

3 bản đồ × 3 tốc độ × (8 seed train + 4 seed val) = **108 episode** (không phải 96). Chỉ thực hiện sau khi smoke thành công:

```powershell
python .\agent4_longitudinal_v316\collect_v316.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --maps S,C,X --train-seeds 170-177 --val-seeds 186-189 --speeds 2,3,4 --out-dir .\agent4_v316_data
```

Nếu ngắt giữa chừng, **chạy lại chính lệnh này thêm `--resume`**, không đổi bất kỳ thông số nào.

## 5. Huấn luyện

```powershell
python .\agent4_longitudinal_v316\train_v316.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --collection .\agent4_v316_data --out-dir .\agent4_v316_model
```

Đầu ra:

- `agent4_v316_model/agent4_v316_longitudinal.joblib`
- `agent4_v316_model/metrics_v316.json`

## 6. Validation closed-loop (72 lượt mô phỏng)

3 bản đồ × 4 val seeds × 3 tốc độ × 2 profile = **72 episode**.

```powershell
python .\agent4_longitudinal_v316\eval_v316.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --model .\agent4_v316_model\agent4_v316_longitudinal.joblib --mode validation --maps S,C,X --seeds 186-189 --speeds 2,3,4 --out .\agent4_v316_validation.json
```

Kết quả: `summary`, `decision.accepted`, chi tiết `episodes` và lỗi. Không chạy final trước khi xem validation. `--resume` hợp lệ khi giữ nguyên config.

## 7. Giới hạn và bảo toàn kết quả

- **Không sửa file thuộc V3.13/V3.15**, và không thay thế model steering.
- **Không lưu weights/dataset vào GitHub** nếu không chủ ý; CSV có thể rất lớn.
- Kịch bản không traffic, chỉ có ba kiểu map, dùng hình học privileged từ MetaDrive. Chưa có bằng chứng mô hình này xử lý ô tô/người đi bộ/đèn giao thông.
- Mô hình được thiết kế fail-closed khi dữ liệu/nhãn không phù hợp; nếu lỗi API, cung cấp `collection_manifest.json` để sửa đúng nguyên nhân.
- Đừng kết luận V3.16 tốt hơn khi chỉ dựa vào offline MAE; phải xem closed-loop.
