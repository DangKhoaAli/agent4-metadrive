# Agent 4 V3.12 — Curved Teacher Diagnostic

**Mục tiêu:** Kiểm tra `IDMPolicy` của MetaDrive như một **giáo viên tham chiếu độc lập** trên các bản đồ `C`, `X`, trước khi thu thập dữ liệu để huấn luyện Agent 4 steering mới.

## Vì sao cần V3.12?

- V3.10 steering đã giữ làn và phục hồi tốt trên `S/SSSSS` nhưng trên `C/X` ở 3 m/s chỉ về đích **1/4** (seed 158–159).
- `rule_lateral` cũng đạt **1/4**, PDD V2 đạt **0/4** trên cùng các cấu hình.
- Vì học sinh gần như tái tạo giáo viên, việc bắt chước giáo viên chưa đủ tốt là lựa chọn không hợp lý.
- Báo cáo V3.11B đã chấm lại và đánh dấu `model_predictions_valid=True` trên cả 4 lượt learned.

**V3.12 không huấn luyện model**, không thay đổi file `.joblib`. MetaDrive điều khiển xe bằng `IDMPolicy` đã được bọc lớp ghi hành động. `env.step([0,0])` chỉ là *placeholder*; chương trình xác nhận IDM thực sự phát lệnh ở **mỗi bước**. Nếu không, báo cáo đánh dấu lỗi thay vì thành công.

## Bố trí thư mục

Giải nén ZIP vào `D:\Học\Tri tue nhan to nang cao\lab\dataset`:

```text
dataset/
  agent4_curved_teacher_v312/
    curve_teacher_v312.py
    geometry_v312.py
    test_v312.py
    README.md
  agent4_v311b_curve_3mps_rescored.json  (nếu có)
```

Yêu cầu: Python, MetaDrive **0.4.3** (môi trường đã chạy các phiên bản trước). Không cần tải PDD hay V3.10 joblib để chạy baseline IDM.

## 1. Kiểm thử offline

```powershell
python agent4_curved_teacher_v312\test_v312.py
```

## 2. Smoke test — `C`, `X`, seed 158

```powershell
python agent4_curved_teacher_v312\curve_teacher_v312.py --maps C,X --seeds 158 --steps 2500 --idm-speed-kmh 10.8 --trace-stride 15 --out ".\agent4_v312_idm_smoke.json"
```

Kết quả: `agent4_v312_idm_smoke.json` và `.csv`. Xem `confirmed_success`, `recorded_actions_aligned`, `status`, `max_speed_mps`, `max_abs_planned_lateral_m`, `trace`.

## 3. Đối chứng cả hai seed (sau smoke test)

```powershell
python agent4_curved_teacher_v312\curve_teacher_v312.py --maps C,X --seeds 158,159 --steps 2500 --idm-speed-kmh 10.8 --trace-stride 15 --reference ".\agent4_v311b_curve_3mps_rescored.json" --out ".\agent4_v312_idm_compare.json"
```

Nếu file tham chiếu có tên `agent4_v311b_curve_3mps_rescored(2).json`, đổi đúng `--reference` hoặc bỏ tham số để chạy IDM độc lập.

`IDMPolicy.NORMAL_SPEED` dùng đơn vị **km/h**, vì thế `10.8 km/h = 3 m/s`. Đây là **tốc độ đặt cho chính sách, không phải tốc độ tối đa cưỡng chế**. Cần kiểm tra tốc độ thực tế trong JSON. Thử IDM ở tốc độ mặc định có thể bổ ích về sau nhưng không được coi là so sánh công bằng ở 3 m/s.

## 4. Chạy tiếp khi bị ngắt

Chạy lại **đúng lệnh cũ**, thêm `--resume`. Báo cáo được lưu sau mỗi episode. Thêm `--overwrite` nếu chủ động muốn ghi đè dữ liệu cũ; không dùng đồng thời `--resume`.

## Dữ liệu ghi lại

- `status`, `arrive_dest`, số bước, quãng đường, route completion, tốc độ thực tế, kết quả chấm thành công.
- Hành động IDM (lái/ga-phanh), kiểm tra số lần `IDM.act` tương ứng số `env.step`.
- Sai số ngang phải-phương tuyến, sai số hướng, khoảng cách tới cuối làn, độ cong proxy từ thay đổi hướng làn trước 8 m.
- Phân biệt `vehicle_lane` và `navigation.current_ref_lanes`, làn tiếp theo và góc hướng chênh tại chuyển đoạn.
- Các bước vượt độ lệch 15/30/50 cm, thay đổi làn, gần mép đường; lưu thêm toàn bộ 55 bước cuối.
- `outcome_comparison` so sánh theo **đúng map/seed** với JSON V3.11B; không đánh đồng quỹ đạo hoặc tốc độ.

## Điều kiện quyết định

**Nếu IDM thành công trên cả `C`, `X`:** kiểm tra trace có sai số lớn không, sau đó mới cân nhắc thu thập demo. **Nếu IDM thất bại trên một tuyến:** kiểm tra xem có lỗi định tuyến hoặc điều kiện bản đồ không, chưa tự động dùng IDM làm giáo viên. **Nếu IDM thành công nhưng tốc độ thực tế khác đáng kể 3 m/s:** so sánh outcome chỉ mang tính định hướng, cần cấu hình thử tương đương tốc độ.

**Giới hạn:** Đường không có traffic; có thông tin hình học tuyến từ simulator; không phải bài kiểm tra tự hành từ camera, không có giá trị chứng nhận an toàn thực tế. Chưa kiểm thử MetaDrive trực tiếp trong môi trường tạo ZIP, chỉ có kiểm thử unit giả lập.

## Tham khảo MetaDrive

- [MetaDrive Action & Policy: IDMPolicy](https://metadrive-simulator.readthedocs.io/en/latest/action.html)
- Mã cũ cùng dự án: `agent4_calibration_v26/probe_idm_action_v26.py` (đã sử dụng `agent_policy=RecordedIDMPolicy`).
