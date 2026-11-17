"""PDD stage-2 diagnostic: physical consistency and *candidate* control timing.

Usage (Windows PowerShell / cmd):
    python verify_pdd_physics_timing.py "D:\\path\\to\\PDD"
    python verify_pdd_physics_timing.py ".\\PDD" --out pdd_physics_timing.json

No third-party dependencies. Does NOT certify temporal alignment or create training labels.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def num(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def med(xs):
    return round(median(xs), 6) if xs else None


def corr(points):
    if len(points) < 30:
        return None
    n = len(points)
    sx = sum(p[0] for p in points) / n
    sy = sum(p[1] for p in points) / n
    xx = sum((x - sx) ** 2 for x, _ in points)
    yy = sum((y - sy) ** 2 for _, y in points)
    if xx * yy < 1e-18:
        return None
    xy = sum((x - sx) * (y - sy) for x, y in points)
    return round(xy / math.sqrt(xx * yy), 4)


def calc_scene(scene, stats, shift_pairs, counts):
    metric = scene / 'metric'
    infop = metric / 'metric_info.json'
    controlp = metric / 'metrics.json'
    if not infop.is_file() or not controlp.is_file():
        counts['missing_files'] += 1
        return
    try:
        infos = json.loads(infop.read_text(encoding='utf-8'))
        records = json.loads(controlp.read_text(encoding='utf-8'))['records']
    except (OSError, ValueError, KeyError, TypeError):
        counts['unreadable_files'] += 1
        return
    if not isinstance(infos, dict) or not isinstance(records, list):
        counts['wrong_schema'] += 1
        return
    ctl = {}
    for r in records:
        c = r.get('control', {}) if isinstance(r, dict) else {}
        if isinstance(c, dict) and type(c.get('step')) is int:
            ctl[c['step']] = c
    ordered = sorted((int(k), v) for k, v in infos.items() if k.isdecimal() and isinstance(v, dict))
    counts['scenarios'] += 1
    counts['telemetry_records'] += len(ordered)
    counts['control_records'] += len(records)
    counts['matching_step5'] += sum(b[0] - a[0] == 5 for a, b in zip(ordered, ordered[1:]))
    counts['non_step5'] += sum(b[0] - a[0] != 5 for a, b in zip(ordered, ordered[1:]))
    for step, info in ordered:
        speed = num(info.get('speed'))
        velocity = info.get('velocity')
        if speed is not None and isinstance(velocity, list) and len(velocity) >= 3:
            xyz = [num(v) for v in velocity[:3]]
            if all(v is not None for v in xyz):
                vnorm = math.sqrt(sum(v*v for v in xyz))
                stats['speed_abs_minus_velocity_norm_mps'].append(abs(speed - vnorm))
        limit = num(info.get('speed_limit'))
        if limit is not None:
            counts[f'speed_limit_raw_{limit:g}'] += 1
        dist = num(info.get('distance_to_front_vehicle'))
        if dist == -1.0:
            counts['front_distance_sentinel_minus_one'] += 1
        c = ctl.get(step)
        if c:
            braking = num(c.get('brake'))
            if braking is not None and braking > 0.05:
                counts['brake_over_005_at_frame'] += 1
    for (s0, i0), (s1, i1) in zip(ordered, ordered[1:]):
        if s1 - s0 != 5:
            continue
        v0, v1 = num(i0.get('speed')), num(i1.get('speed'))
        loc0, loc1 = i0.get('location'), i1.get('location')
        yaw0, yaw1 = None, None
        rot0, rot1 = i0.get('rotation'), i1.get('rotation')
        if isinstance(rot0, list) and len(rot0) >= 3 and isinstance(rot1, list) and len(rot1) >= 3:
            yaw0, yaw1 = num(rot0[2]), num(rot1[2])
        if v0 is None or v1 is None:
            continue
        counts['intervals_with_speeds'] += 1
        if isinstance(loc0, list) and isinstance(loc1, list) and len(loc0) >= 2 and len(loc1) >= 2:
            xy = [num(loc0[0]), num(loc0[1]), num(loc1[0]), num(loc1[1])]
            if all(x is not None for x in xy):
                travel = math.hypot(xy[2] - xy[0], xy[3] - xy[1])
                expected = (v0 + v1) / 2 * 0.25  # Source README: 4 FPS
                if expected > 0.25 and travel < 25:
                    stats['travel_over_speed_times_025s'].append(travel / expected)
        if yaw0 is None or yaw1 is None:
            continue
        dyaw_deg = (yaw1 - yaw0 + 180) % 360 - 180
        if abs(dyaw_deg) > 45:  # avoid teleports / problematic angle samples
            continue
        for offset in (-5, 0, 5):
            steps = range(s0 + offset, s0 + offset + 5)
            actions = [ctl.get(t) for t in steps]
            if any(a is None for a in actions):
                continue
            steer = [num(a.get('steer')) for a in actions]
            throt = [num(a.get('throttle')) for a in actions]
            brake = [num(a.get('brake')) for a in actions]
            if not all(x is not None for x in steer + throt + brake):
                continue
            avs = sum(steer) / 5
            ava = sum(t-b for t,b in zip(throt,brake)) / 5
            if (v0+v1)/2 >= 2.0:  # yaw change near standstill too noisy
                shift_pairs[offset]['steer_vs_next_yaw_deg'].append((avs, dyaw_deg))
            shift_pairs[offset]['a2_vs_next_speed_change_mps'].append((ava, v1 - v0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pdd_root', type=Path)
    ap.add_argument('--out', type=Path, default=Path('pdd_physics_timing.json'))
    args = ap.parse_args()
    root = args.pdd_root.expanduser().resolve()
    if not root.is_dir():
        ap.error(f'PDD directory not found: {root}')
    stats = defaultdict(list)
    shift_pairs = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    for driver in ('driver_01', 'driver_02', 'driver_03'):
        base = root / driver / 'data'
        if not base.is_dir():
            counts[f'missing_{driver}'] += 1
            continue
        for scene in sorted(base.iterdir()):
            if scene.is_dir():
                calc_scene(scene, stats, shift_pairs, counts)
    shifts = {}
    for offset in (-5, 0, 5):
        sp = shift_pairs[offset]
        shifts[str(offset)] = {
            'control_window': f'[telemetry_step{offset:+d}, telemetry_step{offset+4:+d}]',
            'steer_yaw_n': len(sp['steer_vs_next_yaw_deg']),
            'corr_steer_vs_next_yaw': corr(sp['steer_vs_next_yaw_deg']),
            'accel_speed_n': len(sp['a2_vs_next_speed_change_mps']),
            'corr_a2_vs_next_speed_change': corr(sp['a2_vs_next_speed_change_mps']),
        }
    output = {
        'status': 'CANDIDATE_TIMING_DIAGNOSTICS_ONLY',
        'pdd_root': str(root),
        'counts': dict(sorted(counts.items())),
        'speed_vs_velocity': {
            'n': len(stats['speed_abs_minus_velocity_norm_mps']),
            'median_abs_error_mps': med(stats['speed_abs_minus_velocity_norm_mps']),
            'pct_abs_error_over_05_mps': round(100 * sum(x > .5 for x in stats['speed_abs_minus_velocity_norm_mps']) / len(stats['speed_abs_minus_velocity_norm_mps']), 2) if stats['speed_abs_minus_velocity_norm_mps'] else None,
        },
        'position_vs_speed_4fps': {
            'n': len(stats['travel_over_speed_times_025s']),
            'median_travel_to_expected_ratio': med(stats['travel_over_speed_times_025s']),
            'interpretation': 'Near 1 suggests m/s speed and ~0.25s spacing; not a timestamp certificate.'
        },
        'control_windows': shifts,
        'warnings': [
            'Correlation does not establish exact timestamp alignment or true control latency.',
            'The PDD dataset card labels speed_limit as m/s, but raw values should be checked against original collector before unit conversion.',
            'Do not use these diagnostics as training labels. Do not put expert_control_* into model inputs.',
        ],
    }
    args.out.expanduser().resolve().write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'scenarios': counts['scenarios'],
        'telemetry': counts['telemetry_records'],
        'step5_violations': counts['non_step5'],
        'median_speed_velocity_abs_error_mps': output['speed_vs_velocity']['median_abs_error_mps'],
        'median_travel_to_expected_ratio': output['position_vs_speed_4fps']['median_travel_to_expected_ratio'],
        'control_windows': shifts,
    }, indent=2, ensure_ascii=False))
    print(f'Saved: {args.out.expanduser().resolve()}')


if __name__ == '__main__':
    main()
