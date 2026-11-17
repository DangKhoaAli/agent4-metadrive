# Agent 4 V3.6 — Học ga/phanh từ IDM trong MetaDrive

## Mục tiêu

- **Steering:** giữ nguyên model PDD V2 + Student steering V3.5 (`agent4_v35_student.joblib`).
- **Longitudinal:** huấn luyện một model `HistGradientBoostingRegressor` mới dự đoán **giá trị điều khiển dọc có dấu**, `[-1,+1]`; âm tương ứng phanh, dương tương ứng ga theo quy ước MetaDrive.
- **Thu thập:** thực sự áp dụng ga/phanh do `IDMPolicy.act()` sinh ra (giới hạn chuẩn `[-1,1]`), **không lấy bộ giữ tốc độ của V3.5 làm nhãn expert**. V3.5 đã lưu `original_idm_longitudinal` nhưng thực tế áp dụng `governed_longitudinal` — đó không phải dữ liệu rollout ga/phanh IDM thật.
- **Kiểm định:** cùng một model steering V3.5, so sánh `governor_v35` với `learned_long_v36` trên cùng map/seed/desired speed. Khi thử learned longitudinal **không sử dụng hành động governor làm fallback**, không gọi IDM.

Cần **Python 3.10, MetaDrive 0.4.3, numpy, joblib, scikit-learn** trên máy Windows của bạn. Các package V2.6, V2.7, V2.9, V3.0, V3.1, V3.3, V3.4, V3.5 phải ở thư mục con ngang hàng (như hiện tại). Tất cả lệnh bên dưới chạy **từ thư mục `dataset`**, không phải từ bên trong `agent4_longitudinal_v36`.

## 0. Test mã offline

```powershell
python agent4_longitudinal_v36\test_v36.py
```

Kỳ vọng có dòng: `V3.6 OFFLINE TESTS PASS`. Test gồm kiểm tra lớp IDM giả, huấn luyện với dữ liệu tổng hợp, chặn seed leakage, patch đóng/mở controller. **Không thay thế kiểm tra tích hợp trong MetaDrive thực tế.**

## 1. Thu thập native IDM longitudinal (không cần file model PDD khi chạy lệnh này)

```powershell
python agent4_longitudinal_v36\collect_longitudinal_v36.py --maps S,C,X --speeds 2,3,4 --train-seeds 60-71 --val-seeds 72-75 --eval-seeds 76-79 --steps 1800 --out-dir ".\agent4_v36_idm_data"
```

Dữ liệu sẽ ở:

```text
agent4_v36_idm_data/
  train.csv
  val.csv
  manifest.json
```

- Train = seeds **60–71**; validation = **72–75**; eval = **76–79**, không thu thập eval.
- Các seed **0–59** đã dùng trong các thử nghiệm trước bị khóa, không dùng để huấn luyện V3.6.
- Trong episode chuyên gia, desired speed thay đổi có định kỳ (100% → 55% → 100%) và giảm theo khoảng cách tới đích. Chương trình chỉ đặt **mục tiêu tốc độ** cho IDM, còn **chính IDM tạo ga/phanh** và thực hiện.
- `teacher_longitudinal == applied_longitudinal` và mỗi nhãn được ghép với trạng thái **trước** bước mô phỏng đó. Hành động gốc ngoài dải được giới hạn để tương thích action space.
- Chỉ các episode IDM `arrive_dest` mới được thêm vào CSV. Manifest ghi số lượng mẫu phanh, ga, tỉ lệ giới hạn, trạng thái mỗi seed/map/speed. **Kiểm tra `manifest.json` trước khi train.**
- Nếu `train.csv` có ít mẫu phanh, script sẽ báo lỗi thay vì tuyên bố học được phanh. Nếu lỗi thu thập, chọn `--out-dir` khác (công cụ chặn ghi đè bộ dữ liệu hiện tại).

> IDM steering được sử dụng trong thu thập. V3.5 steering được sử dụng trong đánh giá. Đây là một dạng dịch chuyển phân phối trạng thái (distribution shift) cần đo bằng closed-loop.

## 2. Huấn luyện model V3.6 từ dữ liệu mới

```powershell
python agent4_longitudinal_v36\train_longitudinal_v36.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --data-dir ".\agent4_v36_idm_data" --out-model ".\agent4_v36_longitudinal.joblib" --out-metrics ".\agent4_v36_training_metrics.json"
```

Dữ liệu model: 15 PDD features + 6 biến mới gồm lateral, heading, lookahead, **target_speed_mps**, remaining distance, speed error (21 features). Mô hình **không cần chạy PDD để suy luận longitudinal**; chỉ dùng chung schema đầu vào của PDD. Điểm train/val: MAE, RMSE, brake MAE, recall phát hiện phanh, throttle MAE và các nhóm tốc độ-map.

## 3. Đánh giá closed-loop A/B trên seed CHƯA TỪNG dùng

```powershell
python agent4_longitudinal_v36\eval_longitudinal_v36.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --longitudinal ".\agent4_v36_longitudinal.joblib" --maps C,X --seeds 76,77 --speeds 2,3,4 --profiles governor_v35,learned_long_v36 --steps 1800 --out ".\agent4_v36_closed_loop_eval.json"
```

**24 lượt:** 2 bản đồ × 2 seed × 3 tốc độ × 2 bộ điều khiển. Hai cấu hình dùng cùng steering V3.5 và cùng desired-speed planner. Bên `learned_long_v36`, output của model được gửi trực tiếp làm `action[1]`, chỉ giới hạn vào `[-1,1]`. Kết quả báo cáo: trạng thái đích, lệch ngang, vạch đường, số bước phanh, tốc độ tối đa, sai số bám tốc độ, lệnh ga/phanh model dự đoán. Cả hai được kiểm tra bằng chính các guard MetaDrive đã dùng ở V3.5.

**Chỉ nếu kiểm định thành công**, chạy thêm 24 lượt giữ lại trên seed 78–79:

```powershell
python agent4_longitudinal_v36\eval_longitudinal_v36.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --longitudinal ".\agent4_v36_longitudinal.joblib" --maps C,X --seeds 78,79 --speeds 2,3,4 --profiles governor_v35,learned_long_v36 --steps 1800 --out ".\agent4_v36_final_holdout.json"
```

Không sử dụng các seed đã đánh giá để huấn luyện lại; nếu cần điều chỉnh, để dành một nhóm seed mới ngoài 79 làm test cuối cùng.

## 4. File cần gửi lại

1. `agent4_v36_idm_data\manifest.json` — chất lượng dữ liệu, số mẫu phanh thực sự.
2. `agent4_v36_training_metrics.json` — hiệu quả học offline trên train/validation.
3. `agent4_v36_closed_loop_eval.json` — **hiệu quả về đích thực tế** của bộ điều khiển học.

## Giới hạn khoa học

- Chỉ MetaDrive mô phỏng, tốc độ **2–4 m/s**, `traffic_density=0`, dữ liệu đặc quyền từ simulator. **Không chạy trên xe thật.**
- Mô hình dự đoán signed longitudinal action; không có model riêng từng kênh throttle/brake. Trong MetaDrive, cách này tương thích trực tiếp với action 2 chiều `[steer, throttle/brake]`.
- Chưa học quyết định tốc độ mong muốn, tránh va chạm, nhường đường, luật giao thông, hoặc dừng đèn đỏ. Desired-speed planner vẫn theo quy tắc. Không có giao thông nên chưa đủ dữ liệu cho phanh do xe trước.
- Kết quả ngoại suy ngoài các tốc độ/bản đồ trên là **chưa kiểm chứng**. Model V3.6 chưa được đo closed-loop cho tới khi bạn chạy lệnh đánh giá.
