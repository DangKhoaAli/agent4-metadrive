"""Create a PROVISIONAL, state-based behavioral-cloning pilot dataset from PDD.

This is NOT a CARLA-to-MetaDrive validated dataset, and does NOT prove
image/telemetry/action timestamps or waypoint coordinate equivalence.

Inputs: PDD/driver_XX/data/SCENARIO/images/*.jpg,
        PDD/driver_XX/data/SCENARIO/metric/metric_info.json,
        PDD/driver_XX/data/SCENARIO/metric/metrics.json

Outputs: train.csv, val.csv, test.csv, metadata.json.

Windows:
  obs(t) = telemetry for simulator step t = 5 * image_frame_index
  instantaneous label = control[t]
  five-step mean label = mean(control[t], ..., control[t+4])
We only emit complete 5-step windows with the NEXT telemetry frame present.
No expert_control_* fields are allowed as model inputs.

Run:
  python prepare_pdd_pilot.py ".\\PDD" --out ".\\pdd_pilot_output"
Standard-library only. Raw PDD data is never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

FEATURES = [
    'speed_mps',
    'waypoint1_forward_candidate_m', 'waypoint1_lateral_candidate_m',
    'waypoint2_forward_candidate_m', 'waypoint2_lateral_candidate_m',
    'lead_distance_capped_m', 'has_lead_vehicle',
]
TARGETS = [
    'steer_at_t_carla', 'throttle_at_t_carla', 'brake_at_t_carla',
    'a2_at_t_metadrive_candidate',
    'steer_mean_next5_carla', 'throttle_mean_next5_carla',
    'brake_mean_next5_carla', 'a2_mean_next5_metadrive_candidate',
]
IDENTITY = ['driver', 'scenario', 'frame_id', 'sim_step', 'image_path']
FIELDS = IDENTITY + FEATURES + TARGETS


def finite(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def valid_point(point):
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    x, y = finite(point[0]), finite(point[1])
    return (x, y) if x is not None and y is not None else None


def load_scene(scene, driver, counters):
    metric_dir = scene / 'metric'
    try:
        infos = json.loads((metric_dir / 'metric_info.json').read_text(encoding='utf-8'))
        records = json.loads((metric_dir / 'metrics.json').read_text(encoding='utf-8'))['records']
    except (OSError, ValueError, TypeError, KeyError) as exc:
        counters['scene_bad_json_or_missing'] += 1
        return [], f'{driver}/{scene.name}: missing or invalid JSON ({type(exc).__name__})'
    if not isinstance(infos, dict) or not isinstance(records, list):
        counters['scene_invalid_structure'] += 1
        return [], f'{driver}/{scene.name}: unexpected JSON structure'

    images = sorted(
        (p for p in (scene / 'images').glob('*.jpg') if p.stem.isdecimal()),
        key=lambda p: int(p.stem)
    )
    steps = sorted(int(k) for k in infos if str(k).isdecimal())
    # Strict: do not silently pair mismatched indices or holes.
    if (len(images) < 2 or len(images) != len(steps)
        or [int(p.stem) for p in images] != list(range(len(images)))
        or steps != [5 * i for i in range(len(images))]):
        counters['scene_rejected_index_step'] += 1
        return [], f'{driver}/{scene.name}: frame IDs and telemetry steps did not meet strict 5*i rule'

    control = {}
    for entry in records:
        c = entry.get('control') if isinstance(entry, dict) else None
        if isinstance(c, dict) and type(c.get('step')) is int:
            if c['step'] in control:
                counters['duplicate_control_step'] += 1
            else:
                control[c['step']] = c

    output = []
    for idx, frame in enumerate(images[:-1]):
        step = idx * 5
        info = infos.get(str(step))
        control_window = [control.get(step + k) for k in range(5)]
        if not isinstance(info, dict) or any(not isinstance(x, dict) for x in control_window):
            counters['row_missing_info_or_control_window'] += 1
            continue
        speed = finite(info.get('speed'))
        wp1, wp2 = valid_point(info.get('target_point')), valid_point(info.get('target_point_next'))
        lead = finite(info.get('distance_to_front_vehicle'))
        if speed is None or speed < 0 or wp1 is None or wp2 is None or lead is None:
            counters['row_bad_state'] += 1
            continue
        if any(abs(v) > 1000 for v in wp1 + wp2):
            counters['row_implausible_waypoint'] += 1
            continue

        actions = []
        for c in control_window:
            st, th, br = (finite(c.get(key)) for key in ('steer', 'throttle', 'brake'))
            if st is None or th is None or br is None or not (-1 <= st <= 1 and 0 <= th <= 1 and 0 <= br <= 1):
                actions = []
                break
            actions.append((st, th, br))
        if len(actions) != 5:
            counters['row_bad_action'] += 1
            continue
        mean = [sum(x[k] for x in actions) / 5 for k in range(3)]
        st0, th0, br0 = actions[0]
        stm, thm, brm = mean
        has_lead = int(lead >= 0)
        lead_capped = min(lead, 150.0) if has_lead else 150.0
        output.append({
            'driver': driver, 'scenario': scene.name, 'frame_id': idx,
            'sim_step': step, 'image_path': str(frame),
            'speed_mps': speed,
            'waypoint1_forward_candidate_m': wp1[0], 'waypoint1_lateral_candidate_m': wp1[1],
            'waypoint2_forward_candidate_m': wp2[0], 'waypoint2_lateral_candidate_m': wp2[1],
            'lead_distance_capped_m': lead_capped, 'has_lead_vehicle': has_lead,
            'steer_at_t_carla': st0, 'throttle_at_t_carla': th0,
            'brake_at_t_carla': br0, 'a2_at_t_metadrive_candidate': th0 - br0,
            'steer_mean_next5_carla': stm, 'throttle_mean_next5_carla': thm,
            'brake_mean_next5_carla': brm, 'a2_mean_next5_metadrive_candidate': thm - brm,
        })
    counters['rows_kept'] += len(output)
    counters['scenes_accepted'] += 1
    return output, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdd_root', type=Path)
    ap.add_argument('--out', type=Path, default=Path('pdd_pilot_output'))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    root, out = args.pdd_root.expanduser().resolve(), args.out.expanduser().resolve()
    if not root.is_dir():
        ap.error(f'PDD directory not found: {root}')
    out.mkdir(parents=True, exist_ok=True)
    drivers = sorted(p for p in root.glob('driver_*') if (p / 'data').is_dir())
    if not drivers:
        ap.error(f'No PDD driver_XX/data directories under {root}')
    counters, warnings, all_rows = Counter(), [], []
    for driver in drivers:
        for scene in sorted(p for p in (driver / 'data').iterdir() if p.is_dir()):
            rows, error = load_scene(scene, driver.name, counters)
            all_rows.extend(rows)
            if error:
                warnings.append(error)

    if not all_rows:
        raise RuntimeError('No valid rows. Check PDD structure and see warnings.')
    scenarios = sorted(set(r['scenario'] for r in all_rows))
    rng = random.Random(args.seed)
    rng.shuffle(scenarios)
    if len(scenarios) < 5:
        # Still allow testing with a small synthetic PDD folder.
        n_test = int(len(scenarios) >= 3)
        n_val = int(len(scenarios) >= 4)
    else:
        n_test = max(1, round(len(scenarios) * 0.15))
        n_val = max(1, round(len(scenarios) * 0.15))
    test = set(scenarios[:n_test])
    val = set(scenarios[n_test:n_test + n_val])
    train = set(scenarios[n_test + n_val:])
    if not train:
        raise RuntimeError('No training scenarios after split')
    splits = {'train': train, 'val': val, 'test': test}
    split_counts = {}
    for name, allowed in splits.items():
        rows = [r for r in all_rows if r['scenario'] in allowed]
        with (out / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        split_counts[name] = len(rows)

    report = {
        'status': 'PILOT_ONLY_LABEL_ALIGNMENT_AND_TRANSFER_NOT_PROVEN',
        'source_root': str(root), 'drivers': [p.name for p in drivers],
        'rows_total': len(all_rows), 'split_row_counts': split_counts,
        'split_scenarios': {k: sorted(v) for k, v in splits.items()},
        'input_features': FEATURES, 'candidate_training_labels': TARGETS,
        'validation': dict(counters), 'warnings': warnings[:60],
        'critical_limitations': [
            'Frame i -> step 5*i is a well-supported CANDIDATE; original image/action timing is not independently certified.',
            'Waypoint x/y are treated as forward/lateral CANDIDATES, not yet verified against MetaDrive vehicle coordinates.',
            'CARLA steering sign/magnitude and throttle-brake scale are not MetaDrive-calibrated.',
            'No raw speed_limit field is used because its actual units are disputed.',
            'No expert_control_* appears in model inputs; these are expert reference controls, NOT necessarily the recorded human actions.',
            'Lead vehicle distance needs a MetaDrive-side physical adapter; this CSV alone cannot be used as closed-loop agent input.',
            'Train/val/test split is by scenario name (across drivers), not by individual frames.',
            'Mean next5 labels apply a control window over 0.25 seconds; do not conflate with instantaneous controls.',
            'These CSVs are experimental data prep artifacts, not an operational autonomous-driving model.'
        ]
    }
    (out / 'metadata.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'status': report['status'], 'drivers': report['drivers'],
        'scenarios_accepted': counters['scenes_accepted'],
        'rows_total': len(all_rows), 'split_row_counts': split_counts,
        'warnings_count': len(warnings), 'saved_to': str(out),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
