# Agent 4 — hiệu chuẩn tọa độ V2.2 (SHADOW)

Bản này **không** dùng dự đoán của Model điều khiển xe. Không sửa dữ liệu PDD hay trọng số model.

## Vì sao cần bản này?

Hai báo cáo V2.1 cho thấy: CARLA/PDD steer dương tương ứng với rẽ phải; MetaDrive steer dương tương ứng với rẽ trái trên đường thử. Dấu dự kiến: `steer_MetaDrive = -steer_CARLA` (độ lớn chưa hiệu chuẩn).

Trong file V2, `wp_right_m` lấy trực tiếp từ `navigation.get_navi_info()`. MetaDrive cho tọa độ ngang theo hệ local; phải đo trục local +Y có hướng trái hay phải trước khi dùng làm giá trị **right**. Nếu +Y là trái, bản V2.2 đảo dấu `wp1_right_m`, `wp2_right_m`, đồng thời hoán đổi **nhãn** xe trái/phải. Các giá trị khác và thứ tự 15 đặc trưng không đổi.

Đồng thời, checkpoint đầu tiên của MetaDrive thường cách xe vài mét, khác waypoint PDD ~27m. **Chưa thay khoảng nhìn trước một cách giả tạo**; script xuất thông tin đường đi để có cơ sở xây dựng mẫu waypoint theo lộ trình trong bước sau.

## Cài đặt

Giải nén folder `agent4_calibration_v22` ngang hàng với `agent4_pdd_v2` trong `dataset`:

```text
dataset/
├── agent4_pdd_v2/
├── agent4_calibration_v22/
├── metadrive_calibration_v21.json
├── pdd_calibration_v21.json
└── pdd_v2_models/at_t/model_v2.joblib
```

## Chạy

```powershell
cd "D:\Học\Tri tue nhan to nang cao\lab\dataset"
python agent4_calibration_v22\test_coordinate_v22.py
python agent4_calibration_v22\probe_shadow_v22.py --model ".\pdd_v2_models\at_t\model_v2.joblib" --md-report ".\metadrive_calibration_v21.json" --pdd-report ".\pdd_calibration_v21.json" --out ".\agent4_v22_shadow.json"
```

Sau khi chạy gửi `agent4_v22_shadow.json`. Cần xem `lateral_signs_found`, đối chiếu `raw_waypoints` / `corrected_waypoints`, và `route_inspection` trước khi sửa đoạn waypoint/horizon chính thức.

### Kiểm tra steering nhiều seed (nếu cần)

V2.1 có 1 trial lái dương ở seed 0 kết thúc sớm. Có thể kiểm tra lại với xung ngắn:

```powershell
python agent4_calibration_v21\calibrate_metadrive_v21.py --seeds 0,1,2,3 --steer-steps 4 --out ".\metadrive_calibration_v21_recheck.json"
```

Nếu chương trình in cảnh báo, mở JSON để xem `early_end` thay vì tự đổi dấu hoặc hệ số.

**Giới hạn:** `model_steering_sign_only_metadrive_candidate` trong JSON chỉ là đề xuất của phép biến đổi dấu. Không được tự động đưa vào `env.step()` trước khi hoàn thành waypoint lookahead và hiệu chuẩn gain/ga/phanh.
