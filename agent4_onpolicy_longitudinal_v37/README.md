# Agent 4 V3.7 — Learned Longitudinal from On-policy Governor Demonstrations

## Vì sao đổi phương pháp?

V3.6 bắt chước ga/phanh của IDM khi **IDM cũng điều khiển hướng lái**. Khi thay bằng steering V3.5, model V3.6 có hành động ga/phanh mạnh hơn governor và thất bại 4/4 ở 4 m/s. V3.6.2 cho thấy governor 80 bước rồi handoff vẫn thất bại 4/4; governor 160 bước rồi handoff thành công 3/4. **Chưa thể kết luận đã giải quyết được nguyên nhân**.

V3.7 thử phương pháp mới, phù hợp hướng ban đầu của dự án: **behavior cloning từ agent rule-based đã thành công**, thay vì tiếp tục bắt chước IDM không tương thích với chính sách steering đang sử dụng. Nguồn nhãn được ghi rõ:

- **Steering khi thu thập:** PDD V2 + learned residual V3.5, giống khi đánh giá.
- **Longitudinal teacher:** governor V3.5, không phải IDM.
- **Longitudinal student:** HistGradientBoostingRegressor được huấn luyện offline từ dữ liệu trạng thái và lệnh governor *đã áp dụng thực tế*.
- **Khi đánh giá:** model student V3.7 tự xuất `action[1]` trực tiếp, không dùng governor hoặc IDM để điều khiển.
- **Cả hai model vẫn nhận `target_speed_mps` do speed planner lập theo quy tắc**; Agent 4 chưa phải end-to-end autonomous policy.

Đây là **thu thập và huấn luyện model thật**. Tuy nhiên, cần gọi đúng là **mô phỏng hành vi của giáo viên governor**, không gọi là mô phỏng IDM. Nếu hiệu quả chưa vượt baseline, vẫn phải ghi nhận là model chưa tốt.

## Sắp xếp thư mục

Giải nén cạnh `agent4_controller_v30`, `agent4_controller_v31`, `agent4_learned_residual_v33`, `agent4_robustness_v34`, `agent4_multispeed_v35`, `agent4_longitudinal_v36` và toàn bộ các dependencies có sẵn. Lệnh chạy từ thư mục `dataset` (thư mục cha).

Cần Python 3.10, MetaDrive 0.4.3, numpy, scikit-learn, joblib (như các bản trước).

### Bước 0 — Unit tests (không cần MetaDrive)

```powershell
python agent4_onpolicy_longitudinal_v37\test_v37.py
```

### Bước 1 — Thu thập dữ liệu teacher governor chạy dưới steering V3.5

```powershell
python agent4_onpolicy_longitudinal_v37\collect_v37.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --maps S,C,X --speeds 2,3,4 --train-seeds 80-91 --val-seeds 92-95 --eval-seeds 96-99 --steps 1800 --out-dir ".\agent4_v37_governor_data"
```

Output: `agent4_v37_governor_data/train.csv`, `val.csv`, `manifest.json`.

- Chỉ ghi episode **`arrive_dest`** vào CSV, nhưng giữ trạng thái của episode thất bại trong manifest để đánh giá độ phủ.
- Train/val/eval đều không trùng; không sử dụng seed 0–79.
- Thu thập có ga tăng từ trạng thái nghỉ, ga duy trì và phanh khi giảm tốc gần đích. Hiện chưa mô phỏng dừng cho xe phía trước.
- Kiểm tra `rows_by_speed_map`, `success_by_split`, `accepted_brake_steps`, `episodes` trong manifest. Nếu giáo viên thất bại nhiều, huấn luyện có thể thiếu dữ liệu hoặc bị script từ chối.
- Không ghi đè dataset cũ. Muốn thu thập lại, chọn `--out-dir` mới.

### Bước 2 — Huấn luyện model V3.7

```powershell
python agent4_onpolicy_longitudinal_v37\train_v37.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --data-dir ".\agent4_v37_governor_data" --out-model ".\agent4_v37_longitudinal.joblib" --out-metrics ".\agent4_v37_training_metrics.json"
```

Output: model `.joblib` và metrics offline. Đọc riêng `cruise_action_mae` (sai số ở lệnh nhỏ gần 0) bên cạnh MAE/RMSE toàn cục. **Không suy ra tỷ lệ về đích chỉ từ metrics offline.**

### Bước 3 — Closed-loop A/B/C với seed chưa sử dụng

```powershell
python agent4_onpolicy_longitudinal_v37\eval_v37.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --old-longitudinal ".\agent4_v36_longitudinal.joblib" --longitudinal ".\agent4_v37_longitudinal.joblib" --maps C,X --seeds 96,97 --speeds 2,3,4 --profiles governor_v35,learned_long_v36,learned_governor_v37 --steps 1800 --out ".\agent4_v37_closed_loop_eval.json"
```

36 lượt: 2 bản đồ × 2 seed × 3 tốc độ × 3 chính sách longitudinal. Cùng **steering V3.5** cho cả ba. Cột A là teacher governor; B là V3.6 học từ IDM; C là V3.7 học từ governor dưới steering V3.5.

Chỉ sau khi xem bộ seed 96–97 mới đánh giá tiếp seed 98–99. Không dùng seed 96–99 để train hoặc tinh chỉnh một model dự kiến kiểm định trên chính những seed đó.

### Bước 4 — Holdout (nếu kết quả bước 3 khả quan)

```powershell
python agent4_onpolicy_longitudinal_v37\eval_v37.py --pdd ".\pdd_v2_models\at_t\model_v2.joblib" --steering ".\agent4_v35_student.joblib" --old-longitudinal ".\agent4_v36_longitudinal.joblib" --longitudinal ".\agent4_v37_longitudinal.joblib" --maps S,C,X --seeds 98,99 --speeds 2,3,4 --profiles governor_v35,learned_long_v36,learned_governor_v37 --steps 1800 --out ".\agent4_v37_final_holdout.json"
```

## Gửi lại

1. `agent4_v37_governor_data/manifest.json`
2. `agent4_v37_training_metrics.json`
3. `agent4_v37_closed_loop_eval.json`

## Giới hạn và tiêu chí

- `traffic_density=0`, chỉ trên MetaDrive. Chưa kiểm thử với giao thông hoặc xe thật.
- Nhãn ga/phanh từ **governor rule-based**. V3.7 **không phải** model IDM mới và không chứng minh rằng chính sách learned vượt giáo viên.
- Thành công = MetaDrive `arrive_dest`, đồng thời quan sát `out_of_road`, `crash`, lệch làn, bám tốc độ.
- So sánh trước khi cải thiện hoặc tinh chỉnh cần tránh seed leakage.
- Nếu `C76` đặc biệt khó, **không** dùng nó làm tập đánh giá held-out sau khi điều chỉnh theo lỗi C76.
- Unit tests chạy bằng dữ liệu giả lập, **không thay thế MetaDrive integration/closed-loop**.
