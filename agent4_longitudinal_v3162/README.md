# Agent 4 — V3.16.2: On-policy DAgger cho điều khiển ga/phanh

## Mục tiêu

Probe V3.16.1 thất bại: `C/186/2m/s` tăng cảnh báo gần mép từ 26 lên 126, khoảng cách mép tối thiểu còn 0.218 m; `C/189/2m/s` đạt chất lượng nhưng sai số tốc độ cao hơn. Giảm số bước phanh bằng bộ lọc **không đủ** để chứng minh hành vi điều khiển ổn định.

V3.16.2 **không thay đổi** V3.13, V3.15, V3.16 hoặc V3.16.1. Mã này thu thập thêm trạng thái **do V3.16 tự điều khiển ga/phanh** (on-policy) để IDM gắn nhãn *shadow*; sau đó huấn luyện lại model giám sát, không dùng rule governor hay IDM để lái xe ở chế độ AI.

**Cảnh báo:** Mã chỉ chạy MetaDrive với hình học map privileged và không có giao thông (`traffic_density=0`). Đây không phải bảo đảm an toàn trong thực tế.

## Vị trí giải nén

Giải nén `agent4_longitudinal_v3162/` ngay trong thư mục `dataset`, cùng cấp với:

- `agent4_longitudinal_v316/` và các thư mục nguồn V3.13/V3.15 cũ;
- `agent4_v312b_idm_dataset/`;
- `agent4_v313_model/`;
- `agent4_v316_data/` (108 episodes đã thu thập);
- `agent4_v316_model/`.

## 0. Kiểm tra mã

```powershell
python .\agent4_longitudinal_v3162\test_v3162.py
```

Đã chạy **28/28 unit tests PASS** trên mã gốc tại thời điểm đóng gói. Chưa thử MetaDrive runtime vì không có dataset/model hoặc môi trường mô phỏng của bạn trong container kiểm thử.

## 1. Thu thập DAgger TRAIN, chưa chạm vào validation

```powershell
python .\agent4_longitudinal_v3162\collect_v3162.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --actor .\agent4_v316_model\agent4_v316_longitudinal.joblib --maps S,C,X --train-seeds 178-185 --speeds 2,3,4 --out-dir .\agent4_v3162_dagger_data
```

`3 maps × 8 train seeds × 3 speeds = 72 lượt`. Các seed `178-185` thuộc nhóm TRAIN của V3.12B, khác 72 lượt train đầu của V3.16 (`170-177`). Với DAgger, `out_of_road` **có thể xuất hiện**, chỉ dùng những nhãn hợp lệ; **không** đổi trạng thái lỗi thành thành công.

Nếu bị ngắt, chạy lại cùng lệnh với `--resume`. Gửi `agent4_v3162_dagger_data/collection_manifest.json` để kiểm tra chất lượng nhãn và phân bố phanh trước khi huấn luyện.

## 2. Huấn luyện model mới (sau khi xác nhận collection đạt)

```powershell
python .\agent4_longitudinal_v3162\train_v3162.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --actor .\agent4_v316_model\agent4_v316_longitudinal.joblib --v316-collection .\agent4_v316_data --dagger-collection .\agent4_v3162_dagger_data --out-dir .\agent4_v3162_model
```

Nguồn fitting:
- V3.16 shadow IDM TRAIN `170-177` (32,307 hàng hợp lệ);
- DAgger shadow IDM TRAIN `178-185` (số hàng xác định sau khi mô phỏng);
- Anchor IDM original TRAIN (~10,997).

Validation V3.16 `186-189` chỉ được dùng để **đo sai số offline**, không đưa vào `estimator.fit`. Validation này đã được xem kết quả trước đó, nên không được gọi là holdout độc lập.

Đầu ra `agent4_v3162_model/agent4_v3162_longitudinal.joblib`, `metrics_v3162.json`.

## 3. Probe lại hai kịch bản cũ (chỉ khi metrics hợp lệ)

```powershell
python .\agent4_longitudinal_v3162\eval_v3162.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --actor .\agent4_v316_model\agent4_v316_longitudinal.joblib --candidate .\agent4_v3162_model\agent4_v3162_longitudinal.joblib --reference .\agent4_v316_validation.json --maps C --seeds 186,189 --speeds 2 --out .\agent4_v3162_probe_C2.json
```

Hai profile lịch sử RULE/V3.16 được tái dùng từ file cũ sau xác minh provenance; chỉ **2 lần mô phỏng mới** cho V3.16.2.

Bộ lọc chấp nhận yêu cầu: không suy giảm arrival và `quality_pass` ở từng cặp so với cả RULE lẫn V3.16; MAE không vượt `baseline + max(0.05 m/s, 5%)`; AI thật sự phanh. Probe hai tình huống **không thể** tự động chấp nhận version vì cần full validation 36 cặp.

## 4. Validation đầy đủ (chỉ nếu probe có triển vọng)

```powershell
python .\agent4_longitudinal_v3162\eval_v3162.py --legacy-root . --dataset .\agent4_v312b_idm_dataset --baseline .\agent4_v313_model\agent4_v313_steering.joblib --actor .\agent4_v316_model\agent4_v316_longitudinal.joblib --candidate .\agent4_v3162_model\agent4_v3162_longitudinal.joblib --reference .\agent4_v316_validation.json --maps S,C,X --seeds 186-189 --speeds 2,3,4 --out .\agent4_v3162_validation.json
```

36 lần mô phỏng **mới** cho candidate, 72 lượt đối chứng tham chiếu được xác nhận. Tập kiểm thử cuối nên sử dụng seeds **>=202**, hoàn toàn mới; không dùng `194-201` đã kiểm thử V3.15. Đừng chạy final trước khi đóng băng candidate và xem kết quả full validation.

## Một số giới hạn quan trọng

- Các trạng thái DAgger được tạo bằng **model V3.16 cũ**, không phải model mới V3.16.2; đây là một vòng DAgger, chưa phải lặp đến hội tụ.
- Việc on-policy collection có thể dẫn đến `out_of_road`; mô hình không được coi là đạt an toàn chỉ nhờ cải thiện offline MAE.
- `teacher_calls = 0` trong đánh giá nghĩa là không gọi IDM; khi thu thập TRAIN, shadow IDM gọi để gắn nhãn nhưng **không trực tiếp điều khiển**.
- Mô hình vẫn dùng target-speed schedule đặt sẵn, chưa tự nhận biết biển tốc độ hoặc ra quyết định tốc độ dựa trên giao thông.
- Khi huấn luyện thực tế nên dùng `scikit-learn` cùng major/minor version với môi trường load model, và chỉ mở file `joblib` từ nguồn tin cậy.
