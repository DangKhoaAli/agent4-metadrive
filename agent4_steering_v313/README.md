# Agent 4 V3.13 — Learned Steering from V3.12B IDM Dataset

## Dữ liệu được xác nhận

Từ `manifest(8).json` của người dùng: **216/216 episode IDM thành công**, **94.184 dòng clean** (train 63.445 / val 15.367 / test 15.372), ba map `S,C,X`, tốc độ 2–4 m/s. Tập `X` có **15/72 episode không có đoạn cong thực tế**. Tất cả 216 episode có nhãn điều khiển được đối chiếu, không có lỗi runtime theo manifest.

**Cảnh báo:** `manifest.json` chỉ có thống kê + đường dẫn. Để thực sự huấn luyện trên dữ liệu thật, phải có thư mục `agent4_v312b_idm_dataset/clean/` chứa các CSV từ máy đã chạy MetaDrive. Gói ZIP này **không chứa model `.joblib` đã huấn luyện**, vì các CSV thật chưa được cung cấp trong cuộc trò chuyện. Kết quả các unit test chỉ sử dụng dữ liệu giả lập.

## 0. Thư mục và môi trường

Giải nén các gói V3.12B và V3.13 **cùng cấp** trong thư mục `dataset`:

```text
dataset/
  agent4_idm_dataset_v312b/
    collect_v312b.py
    geometry_v312.py
    validate_v312b.py
  agent4_steering_v313/
    train_v313.py
    eval_closed_loop_v313.py
    features_v313.py
    test_v313.py
  agent4_v312b_idm_dataset/
    manifest.json
    clean/train/S/*.csv
    clean/train/C/*.csv
    clean/train/X/*.csv
    clean/val/... 
    clean/test/...
    audit/...
```

Cần `numpy`, `scikit-learn`, `joblib` trong cùng Python/venv; để **đánh giá closed-loop** cần MetaDrive 0.4.3.

```powershell
python -m pip install numpy scikit-learn joblib
```

## 1. Kiểm tra dữ liệu và code

```powershell
python agent4_idm_dataset_v312b\validate_v312b.py --dataset ".\agent4_v312b_idm_dataset"
python agent4_steering_v313\test_v313.py
```

Validator sẽ báo lỗi nếu file CSV bị thiếu hoặc dữ liệu audit/clean không nhất quán. Unit test V3.13 chạy offline, không thay thế kiểm thử MetaDrive thật.

## 2. Huấn luyện V3.13 (steering only)

```powershell
python agent4_steering_v313\train_v313.py --dataset ".\agent4_v312b_idm_dataset" --models ridge,hgb,extra --out-dir ".\agent4_v313_model"
```

Các đầu ra:

- `agent4_v313_model/agent4_v313_steering.joblib`: model steering dự đoán trực tiếp trong hệ dấu của MetaDrive, **không phải residual PDD**, **không cần PDD .joblib**.
- `agent4_v313_model/metrics_v313.json`: MAE, RMSE, P95 error; tách theo `S,C,X`, tốc độ, `CircularLane`/non-circular, và so với baseline luôn steering=0.

**Các phương pháp được thử:** Ridge, HistGradientBoosting, ExtraTrees; sample weights cân bằng theo episode. Chọn **duy nhất bằng validation** (186–189), ưu tiên C/X 2× S, rồi đánh giá **một lần** trên tập test (190–193). Không dùng seed, tên bản đồ, tọa độ tuyệt đối, hậu trạng thái, hành động IDM thô hay output giáo viên làm input. Dữ liệu train chỉ dùng 170–185.

**Đầu vào 12 chiều:** speed, lateral error, heading error, khoảng cách đến cuối lane, curvature proxy, lookahead heading 5/10/20 m, cờ khả dụng các lookahead, cờ CircularLane. Dùng đúng bộ trích xuất của collector V3.12B khi đánh giá.

**Lưu ý:** clean CSV loại các mẫu không có lookahead 5/10 m. Khi model tự lái và gần cuối tuyến, các horizon có thể không khả dụng: code sẽ mã hóa `available=0` với heading 0, đánh dấu số bước đó trong báo cáo. **Những trạng thái này không xuất hiện trong dữ liệu huấn luyện clean và cần xem là ngoài phân phối**. Không có fallback sang IDM; kết quả closed-loop sẽ quyết định hướng xử lý tiếp.

## 3. Closed-loop smoke test trên VALIDATION (chưa đụng test)

```powershell
python agent4_steering_v313\eval_closed_loop_v313.py --model ".\agent4_v313_model\agent4_v313_steering.joblib" --maps C,X --seeds 186 --speeds 3 --steps 2500 --out ".\agent4_v313_val_smoke.json"
```

Đánh giá chỉ dùng **model steering vừa huấn luyện**; throttle/brake do bộ điều khiển `P` cố định với `kp=0.23`, giới hạn ±0.4. Không gọi IDM, không dùng model PDD, không gọi giáo viên. Các seed 186–189 đã dùng validation để chọn model, nên chỉ dùng cho smoke/chỉnh thuật toán, **không phải kết quả cuối**.

Sau smoke, kiểm tra `confirmed_success`, `quality_pass`, `max_abs_planned_lateral_m`, `min_edge_clearance_proxy_m`, `near_edge_warning_steps`, `missing_lookahead_5_10_steps`, `errors`. Cờ `confirmed_success` yêu cầu xe về đích và mọi bước model có lệnh đã áp dụng khớp. `quality_pass` còn đòi lateral ≤0.6m, heading ≤0.12rad, clearance proxy ≥0.3m, không cảnh báo gần mép hay chạm vạch vàng — chuẩn so sánh cùng V3.12B.

## 4. Kiểm thử cuối trên tập TEST (chỉ sau khi smoke ổn)

```powershell
python agent4_steering_v313\eval_closed_loop_v313.py --model ".\agent4_v313_model\agent4_v313_steering.joblib" --maps S,C,X --seeds 190-193 --speeds 2,3,4 --steps 2500 --out ".\agent4_v313_closed_loop_test.json"
```

36 episode. Báo cáo JSON + CSV lưu sau từng episode. Nếu bị ngắt, lặp lại **chính xác cùng lệnh** thêm `--resume` (model SHA256/config phải khớp). Không điều chỉnh theo test để báo cáo là unseen holdout; nếu model chưa đạt, dùng validation để chỉnh, và sau đó cần seed test **hoàn toàn mới**.

## 5. Thành công nghĩa là gì?

- **Offline MAE thấp**: bắt chước nhãn trong trạng thái của IDM; **không bảo đảm xe về đích**.
- **Closed-loop `arrive_dest` + `quality_pass`**: học được steering đủ tốt trên seed/map/tốc độ cụ thể trong MetaDrive mô phỏng.
- **End-to-end learned autonomy**: *chưa đạt*, vì throttle/brake còn rule-based và quan sát hình học bản đồ là privileged.
- **Chạy xe vật lý an toàn**: *không kết luận*, chưa có cảm biến/traffic/thử nghiệm thực tế.

## 6. Thông tin cần gửi lại

Sau huấn luyện: `agent4_v313_model/metrics_v313.json`. Sau smoke: `agent4_v313_val_smoke.json`. Nếu muốn mình trực tiếp huấn luyện trên dữ liệu thực tại đây, hãy ZIP **toàn bộ `clean/` + `manifest.json`**, không cần `audit/` (validator đầy đủ mới cần cả audit).
