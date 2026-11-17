# Agent 4 — V2.6: So sánh action IDM / Model PDD (shadow)

## Tại sao cần bước này?

V2.5 kiểm tra được 6 lượt IDMPolicy đều về đích trên map C/X, có mẫu ở đường cong và giao lộ. Tuy nhiên *model PDD chưa lái*. Vì steering model đôi lúc khá nhỏ, cần so sánh action của model và action IDM **ở cùng trạng thái/timestep**. Bộ V2.6 làm việc này mà **không thay action điều khiển của IDM**.

## Cài đặt

Giải nén `agent4_calibration_v26/` vào thư mục `dataset/`, ngang cấp với:

- `agent4_pdd_v2/`
- `agent4_calibration_v22/`
- `agent4_calibration_v23/`
- `agent4_calibration_v24/`
- `pdd_v2_models/at_t/model_v2.joblib`

Máy của người dùng: Python 3.10, MetaDrive 0.4.3, numpy, joblib và sklearn phù hợp với model đã tạo.

## Chạy

```powershell
cd "D:\Học\Tri tue nhan to nang cao\lab\dataset"
python agent4_calibration_v26\test_action_v26.py
python agent4_calibration_v26\probe_idm_action_v26.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps "C,X" --seeds "0,1,2" --steps 450 --out ".\agent4_v26_idm_action_probe.json"
```

## Cách hoạt động

- `RecordedIDMPolicy` kế thừa trực tiếp `IDMPolicy`, gọi `super().act()` và **chỉ lưu lại action vừa tạo**. Không thay đổi cách lái.
- Trước mỗi `env.step` tạo observation, waypoint theo làn và dự đoán của model (PDD +right, MetaDrive sign-only -steer).
- Sau chính `env.step` đó, lấy action vừa tính của IDM; kiểm tra `record_count` tăng **đúng một** để tránh so nhầm timestep.
- Thống kê MAE steering và longitudinal, tương quan, tỷ lệ đồng dấu trên các hành động đủ lớn, và hệ số least-squares *chỉ tham khảo*.
- Nếu không đủ 25/45 m đường phía trước thì **không so sánh**, nhưng IDM vẫn chạy đến đích.
- Chương trình không chứa lời gọi `env.step()` với action của model.

## Giới hạn cần tôn trọng

- Hành động IDM là *reference* tại MetaDrive, không phải nhãn PDD; khác policy, tốc độ, dynamics, horizon. Chỉ so sánh không thể chứng minh model đúng/sai.
- `gain_least_squares_through_zero_DIAGNOSTIC_ONLY` **không được áp dụng** cho xe. Nó có thể bị lệch bởi target policy, steering offset, hoặc các mẫu gần 0.
- Dữ liệu không có giao thông (`traffic_density=0.0`), không hiệu chuẩn phanh trong tình huống khẩn cấp.
- Waypoint còn phụ thuộc `nearest-continuous lane`, chưa chứng minh đúng ngữ nghĩa làn bắt buộc qua mọi giao lộ.
- Model không được đưa vào closed-loop; cần thêm kiểm tra dynamics và chương trình thử nghiệm có giới hạn sau khi phân tích báo cáo V2.6.

Kiểm tra trước phát hành: cú pháp và unit test toán học. Chưa chạy MetaDrive thật ở môi trường tạo code, nên báo cáo từ máy người dùng mới là nguồn xác nhận cuối cùng.
