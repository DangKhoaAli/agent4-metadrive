"""Kiểm tra ghép ảnh -> telemetry -> control cho dữ liệu PDD.

Không huấn luyện model và KHÔNG khẳng định đồng bộ thời gian chỉ từ ID.
Ví dụ (Windows):
  python verify_pdd_alignment.py "D:\\Học\\Tri tue nhan to nang cao\\lab\\dataset\\PDD"
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def numbered(folder, extension):
    items = {}
    if folder.is_dir():
        for p in folder.glob('*' + extension):
            stem = p.name[:-len(extension)]
            if stem.isdecimal():
                items[int(stem)] = p
    return items


def audit(scene):
    images = numbered(scene / 'images', '.jpg')
    boxes = numbered(scene / 'boxes', '.json.gz')
    infos = json.loads((scene / 'metric' / 'metric_info.json').read_text(encoding='utf-8'))
    controls = json.loads((scene / 'metric' / 'metrics.json').read_text(encoding='utf-8')).get('records', [])
    info_keys = sorted(int(k) for k in infos if str(k).isdecimal())
    control_map = {}
    duplicates = 0
    for item in controls:
        if not isinstance(item, dict) or not isinstance(item.get('control'), dict):
            continue
        c = item['control']
        step = c.get('step')
        if isinstance(step, int) and not isinstance(step, bool):
            duplicates += step in control_map
            control_map[step] = c
    img_keys = sorted(images)
    info_gaps = [b-a for a,b in zip(info_keys, info_keys[1:])]
    common_gap = Counter(info_gaps).most_common(1)[0][0] if info_gaps else None
    # Giả thuyết A: ảnh thứ i tương ứng với bản ghi telemetry thứ i,
    # và telemetry key là step tương ứng trong metrics.json.
    candidate_count = min(len(img_keys), len(info_keys))
    candidates = [(img_keys[i], info_keys[i]) for i in range(candidate_count)]
    valid_pairs = [(frame,step) for frame,step in candidates if step in control_map and frame in boxes]
    coverage = len(valid_pairs) / max(len(img_keys), 1)
    sample_indices = sorted({0, len(valid_pairs)//2, len(valid_pairs)-1}) if valid_pairs else []
    examples = []
    for idx in sample_indices:
        frame, step = valid_pairs[idx]
        mi = infos[str(step)]
        c = control_map[step]
        examples.append({
            'image_file': f'{frame}.jpg',
            'box_file': f'{frame}.json.gz',
            'telemetry_key': step,
            'control_step': c.get('step'),
            'speed': mi.get('speed'),
            'target_point': mi.get('target_point'),
            'steer': c.get('steer'),
            'throttle': c.get('throttle'),
            'brake': c.get('brake'),
        })
    sequential_images = img_keys == list(range(len(img_keys)))
    uniform_gap = all(g == common_gap for g in info_gaps) if info_gaps else True
    return {
        'driver': scene.parent.parent.name,
        'scenario': scene.name,
        'images': len(img_keys), 'boxes': len(boxes),
        'telemetry_records': len(info_keys), 'control_records': len(controls),
        'image_ids_start_at_zero_and_contiguous': sequential_images,
        'telemetry_first_keys': info_keys[:12],
        'telemetry_last_keys': info_keys[-3:],
        'most_common_telemetry_step_gap': common_gap,
        'telemetry_gap_counts': dict(Counter(info_gaps)),
        'telemetry_gaps_uniform': uniform_gap,
        'control_duplicate_steps': duplicates,
        'proposed_image_telemetry_control_triplets': len(valid_pairs),
        'proposed_triplet_coverage': round(coverage, 6),
        'missing_control_for_telemetry': [k for k in info_keys if k not in control_map][:10],
        'examples': examples,
        'status': 'CANDIDATE_ONLY_not_timestamp_verified' if coverage == 1.0 else 'INCOMPLETE_CHECK',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdd_root', type=Path)
    parser.add_argument('--report', type=Path, default=Path('pdd_alignment_v2.json'))
    args = parser.parse_args()
    root = args.pdd_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Không tìm thấy thư mục: {root}')
    rows = []
    for driver in ('driver_01', 'driver_02', 'driver_03'):
        data_dir = root / driver / 'data'
        if not data_dir.is_dir():
            print('Thiếu thư mục:', data_dir)
            continue
        for scene in sorted(x for x in data_dir.iterdir() if x.is_dir()):
            try:
                x = audit(scene)
                rows.append(x)
                print(f"{driver}/{scene.name:36s} img={x['images']:4d} "
                      f"telemetry-gap={x['most_common_telemetry_step_gap']} "
                      f"triplets={x['proposed_image_telemetry_control_triplets']:4d} "
                      f"coverage={x['proposed_triplet_coverage']:.1%}")
            except Exception as ex:
                print(f'ERROR {driver}/{scene.name}: {type(ex).__name__}: {ex}')
                rows.append({'driver': driver, 'scenario': scene.name, 'error': f'{type(ex).__name__}: {ex}'})
    valid = [x for x in rows if 'error' not in x]
    report = {
        'important': 'PDD image frame index may differ from simulator step. Index-based candidate mapping requires validation against original capture code or time metadata before training.',
        'summary': {
            'scenarios': len(rows),
            'scenarios_with_read_errors': len(rows) - len(valid),
            'images': sum(x['images'] for x in valid),
            'proposed_triplets': sum(x['proposed_image_telemetry_control_triplets'] for x in valid),
            'all_coverage_100pct': bool(valid) and all(x['proposed_triplet_coverage'] == 1.0 for x in valid),
            'telemetry_gap_distribution': dict(Counter(x['most_common_telemetry_step_gap'] for x in valid)),
            'scenarios_with_nonuniform_telemetry_gaps': sum(not x['telemetry_gaps_uniform'] for x in valid),
            'scenarios_with_noncontiguous_image_names': sum(not x['image_ids_start_at_zero_and_contiguous'] for x in valid),
        },
        'scenarios': rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=== SUMMARY ===')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
    print('LƯU Ý: 100% coverage chỉ xác nhận phép ghép theo step tồn tại; không chứng minh timestamp ảnh khớp.')
    print('Report:', args.report.resolve())


if __name__ == '__main__':
    main()
