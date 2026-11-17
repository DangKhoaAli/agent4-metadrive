# Agent 4 — Waypoint calibration V2.4 (lane vs road center, shadow only)

## Vì sao cần V2.4?

V2.3 trên map `S` có 320/320 waypoint hợp lệ, nhưng **waypoint lấy giữa cả con đường (road-wide center)**. Trên seed 0, ego ở `y=7.0m`, waypoint tại `y=3.5m`. Điều này khiến target lệch 3.5m dù đường thẳng. PDD `target_point` thường gần làn đường cần đi (trung vị ngang ~0.15m), vì vậy cần so sánh với mục tiêu **tâm làn xe đang đi (own-lane center)**, không chỉ kiểm tra hình học của tâm đường.

Phiên bản này so sánh song song:

1. **Road-center (V2.3):** lấy tâm toàn bộ các làn của tuyến tại 25/45m.
2. **Own-lane (V2.4 experimental):** chọn làn gần ego trên road đầu, tiếp tục trên các road theo khoảng cách gần nhất giữa cuối/đầu làn; phát hiện đứt đoạn >2m, không che lỗi.

**Lưu ý:** Lựa chọn làn theo tính liên tục **không phải** lập kế hoạch chuyển làn để rẽ vào đích; tại giao lộ/merge, làn phù hợp phụ thuộc tuyến đích, không chỉ hình học. Cả hai kiểu chỉ là ứng viên để kiểm tra.

## Cài đặt và thư mục

Giải nén `agent4_calibration_v24.zip` vào thư mục `dataset/`, giữ cấu trúc:

```text
dataset/
  agent4_pdd_v2/
  agent4_calibration_v22/
  agent4_calibration_v23/
  agent4_calibration_v24/
  pdd_v2_models/at_t/model_v2.joblib
```

Python 3.10, MetaDrive 0.4.3, joblib, numpy. Các thư viện này đã có trên máy nếu V2.3 chạy được.

## Chạy thử

```powershell
cd "D:\Học\Tri tue nhan to nang cao\lab\dataset"
python agent4_calibration_v24\test_lane_v24.py
python agent4_calibration_v24\probe_lane_v24.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --maps "S,C,X" --seeds "0,1,2" --steps 60 --out ".\agent4_v24_lane_probe.json"
```

- `S` = straight, `C` = circular/curved road, `X` = intersection (các ID block được hỗ trợ trong hướng dẫn MetaDrive; tạo bản đồ cụ thể còn phụ thuộc seed và phiên bản).
- Có lỗi trên map `C` hoặc `X` thì chương trình vẫn ghi lỗi vào JSON, **không mặc định là hợp lệ**.
- Ở đường cong/giao lộ, action tham chiếu `[0.0,0.28]` (steering=0) có thể làm xe ra đường; lúc đó cần đọc `reference_episode_end` trước khi kết luận về adapter.
- Không dùng action model để lái. Không chuyển đổi hệ số ga/phanh. Không sửa hoặc huấn luyện lại model.

## File báo cáo

`agent4_v24_lane_probe.json`, trong `results[].summary`:

- `n_samples`, `n_usable`: số mẫu và số mẫu đủ chuẩn để so sánh.
- `n_own_lane_geometry_discontinuous`: số mẫu có liên kết làn bị gián đoạn (cờ quan trọng).
- `n_own_lane_route_incomplete`: tuyến còn lại không đủ chiều dài nhìn trước.
- `road_wp1_right_m`, `lane_wp1_right_m`: phân bố độ lệch ngang của hai ứng viên.
- `road_vs_lane_wp1_right_abs_m`: mức khác biệt mục tiêu ngang.
- `steering_prediction_abs_change`: ảnh hưởng đến dự đoán model; **không phải thước đo khả năng điều khiển**.
- `errors`, `invalid_examples`: lỗi API/vật lý cần kiểm tra.

## Phạm vi kiểm định hiện tại

Đã `py_compile` và kiểm tra bằng hình học mẫu trên máy tạo mã. **Chưa thể thử với MetaDrive 0.4.3 trực tiếp trong môi trường tạo mã này.** Không thay đổi `agent4_pdd_v2` hay `model_v2.joblib` của bạn. Không khuyến nghị chuyển sang closed-loop trước khi báo cáo các map S/C/X được kiểm tra.
