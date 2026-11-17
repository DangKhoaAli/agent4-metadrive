"""Augment the ORIGINAL PDD pilot split with physical scene features (no model leakage).

Inputs:
    pdd_root:       ./PDD
    --pilot:        ./pdd_pilot_output (original train/val/test CSVs)
Output:
    --out:          ./pdd_v2_output (new CSVs and metadata_v2.json)

Retains existing split/targets. NEVER changes raw PDD. This is a PROVISIONAL
cross-simulator contract; training labels and transfer remain unverified.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from shared_adapter_v2 import FEATURES, AdapterDataError, pdd_features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pdd_root', type=Path)
    ap.add_argument('--pilot', type=Path, default=Path('pdd_pilot_output'))
    ap.add_argument('--out', type=Path, default=Path('pdd_v2_output'))
    ap.add_argument('--box-y-sign', type=int, choices=[-1, 1], default=1)
    ap.add_argument('--waypoint-y-sign', type=int, choices=[-1, 1], default=1)
    args = ap.parse_args()
    root, pilot, out = (p.expanduser().resolve() for p in (args.pdd_root, args.pilot, args.out))
    if not root.is_dir():
        ap.error(f'PDD root not found: {root}')
    meta_path = pilot / 'metadata.json'
    if not meta_path.is_file():
        ap.error(f'Pilot metadata missing: {meta_path}')
    pilot_meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
    if pilot_meta.get('status') != 'PILOT_ONLY_LABEL_ALIGNMENT_AND_TRANSFER_NOT_PROVEN':
        ap.error('Pilot metadata status unexpected. Do not mix with another dataset.')
    out.mkdir(parents=True, exist_ok=True)

    # Cache each scene's telemetry JSON (63 files), not all images or boxes.
    cache = {}
    totals, split_counts = Counter(), {}
    examples = []
    issue_samples = []
    feature_values = defaultdict(list)
    extras = FEATURES[:]  # speed exists already in pilot, overwritten consistently.
    for split in ('train', 'val', 'test'):
        src = pilot / f'{split}.csv'
        if not src.is_file():
            ap.error(f'Missing pilot split: {src}')
        dst = out / f'{split}.csv'
        with src.open('r', encoding='utf-8-sig', newline='') as inp, dst.open('w', encoding='utf-8-sig', newline='') as output:
            reader = csv.DictReader(inp)
            assert reader.fieldnames is not None
            fields = list(dict.fromkeys(reader.fieldnames + extras))
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                totals['rows_seen'] += 1
                driver, scene = row['driver'], row['scenario']
                frame_id = int(row['frame_id'])
                step = int(row['sim_step'])
                if step != frame_id * 5:
                    totals['bad_frame_step'] += 1
                    issue_samples.append(f'{driver}/{scene}/{frame_id}: expected step {frame_id*5}, got {step}')
                    continue
                scene_root = root / driver / 'data' / scene
                key = (driver, scene)
                try:
                    if key not in cache:
                        cache[key] = json.loads((scene_root / 'metric' / 'metric_info.json').read_text(encoding='utf-8-sig'))
                    info = cache[key][str(step)]
                    with gzip.open(scene_root / 'boxes' / f'{frame_id}.json.gz', mode='rt', encoding='utf-8') as f:
                        boxes = json.load(f)
                    features = pdd_features(info, boxes, box_y_sign=args.box_y_sign,
                                            waypoint_y_sign=args.waypoint_y_sign)
                    # Assert speed consistency between old and new CSV.
                    if abs(float(row['speed_mps']) - features['speed_mps']) > 0.001:
                        raise AdapterDataError('Old/new speed discrepancy')
                    row.update(features)
                    writer.writerow(row)
                    totals['rows_kept'] += 1
                    totals[f'{split}_rows'] += 1
                    for k, v in features.items():
                        feature_values[k].append(v)
                    if len(examples) < 6:
                        examples.append({'driver': driver, 'scenario': scene, 'frame_id': frame_id,
                                         'features': features})
                except (OSError, ValueError, TypeError, KeyError, IndexError, AdapterDataError) as ex:
                    totals['rows_rejected'] += 1
                    if len(issue_samples) < 30:
                        issue_samples.append(f'{driver}/{scene}/{frame_id}: {type(ex).__name__} {ex}')
        split_counts[split] = totals[f'{split}_rows']

    def stats(values):
        if not values:
            return None
        vals = sorted(values)
        return {'min': vals[0], 'p10': vals[int(.1 * (len(vals)-1))],
                'median': vals[int(.5 * (len(vals)-1))],
                'p90': vals[int(.9 * (len(vals)-1))], 'max': vals[-1]}

    report = {
        'status': 'V2_PROVISIONAL_SHARED_FEATURES_NOT_CLOSED_LOOP_VALIDATED',
        'pdd_root': str(root),
        'pilot_dir': str(pilot),
        'split_counts': split_counts,
        'feature_order': FEATURES,
        'target_label_columns': pilot_meta['candidate_training_labels'],
        'split_scenarios': pilot_meta['split_scenarios'],
        'coordinate_convention': '+x forward, +y right of ego; checkpoint vectors radially capped at 50m',
        'pdd_box_y_sign': args.box_y_sign,
        'pdd_waypoint_y_sign': args.waypoint_y_sign,
        'distance_no_object': '60m cap + has_X flag 0',
        'object_distance': 'Distance to object center, not bumper clearance',
        'counts': dict(totals),
        'feature_stats': {k: stats(v) for k, v in feature_values.items()},
        'examples': examples,
        'rejected_examples': issue_samples,
        'limitations': [
            'Mapping frame i -> 5*i is backed by timing correlations, but NOT certified by original collector.',
            'PDD boxes x/y are documented ego-relative; side sign requires visual/local verification.',
            'PDD and MetaDrive use different route checkpoint placement; physical units match but semantics are not identical.',
            'The object speed field is interpreted as m/s by dataset card; verify it against source if using closing speed.',
            'Closing speed uses speed magnitudes, not relative projected velocities.',
            'Stationary obstacles, traffic lights, priority rules, and cyclists are not fully modeled.',
            'Model outputs are CARLA control candidates; a2=(throttle-brake) is not dynamics-calibrated.',
            'Training split remains by scenario type across three drivers.',
            'This preprocessing is not evidence that Agent 4 will reach destinations.',
        ],
    }
    dest = out / 'metadata_v2.json'
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('V2 rows:', split_counts, '| Rejected:', totals.get('rows_rejected', 0))
    print('Has front vehicle:', totals.get('front', '(see feature_stats.has_front_vehicle)'))
    print('Saved:', dest)
    if issue_samples:
        print('First issues:', issue_samples[:4])
    if any(v == 0 for v in split_counts.values()):
        raise RuntimeError('One split has zero V2 rows; inspect metadata_v2.json')
    if totals.get('rows_rejected', 0) > max(1, totals['rows_seen'] // 100):
        raise RuntimeError('More than 1% rows rejected: inspect metadata_v2.json before training')
    print('CAUTION: semantic compatibility and closed-loop driving have NOT been validated.')


if __name__ == '__main__':
    main()
