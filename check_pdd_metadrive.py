"""Audit local PDD files and probe installed MetaDrive (if available).

Usage:
    python check_pdd_metadrive.py /path/to/PDD
    python check_pdd_metadrive.py D:\\data\\PDD --no-metadrive

Output: pdd_audit.json in current working directory.
No packages are required for the PDD audit.
"""
import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path


def safe_json(path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


def numeric_files(folder, suffix):
    out = {}
    if folder.is_dir():
        for f in folder.glob(f'*{suffix}'):
            name = f.name[:-len(suffix)]
            if name.isdecimal():
                out[int(name)] = f
    return out


def is_valid_action(c):
    try:
        steer, throttle, brake = (float(c[x]) for x in ('steer', 'throttle', 'brake'))
        return (math.isfinite(steer) and math.isfinite(throttle) and math.isfinite(brake)
                and -1 <= steer <= 1 and 0 <= throttle <= 1 and 0 <= brake <= 1)
    except (ValueError, KeyError, TypeError):
        return False


def audit_scenario(driver, scenario):
    photos = numeric_files(scenario / 'images', '.jpg')
    boxes = numeric_files(scenario / 'boxes', '.json.gz')
    raw_info, err_info = safe_json(scenario / 'metric' / 'metric_info.json')
    raw_actions, err_actions = safe_json(scenario / 'metric' / 'metrics.json')
    infos = raw_info if isinstance(raw_info, dict) else {}
    records = raw_actions.get('records', []) if isinstance(raw_actions, dict) else []
    if not isinstance(records, list):
        records = []

    image_ids = set(photos)
    box_ids = set(boxes)
    info_ids = {int(k) for k in infos if str(k).isdecimal()}
    control_map = {}
    all_controls = []
    duplicates = 0
    for record in records:
        c = record.get('control', {}) if isinstance(record, dict) else {}
        if not isinstance(c, dict):
            continue
        all_controls.append(c)
        step = c.get('step')
        if isinstance(step, int) and not isinstance(step, bool):
            if step in control_map:
                duplicates += 1
            control_map[step] = c

    box_matches = len(image_ids & box_ids)
    info_matches = len(image_ids & info_ids)
    action_matches = len(image_ids & set(control_map))
    n_images = len(image_ids)
    sample_key = min(image_ids) if image_ids else None
    sample_info = infos.get(str(sample_key), {}) if sample_key is not None else {}
    if not isinstance(sample_info, dict):
        sample_info = {}
    sample_action = control_map.get(sample_key)
    aligned_count = len(image_ids & info_ids & set(control_map))
    alignment = ('step_and_frame_ids_match' if n_images and aligned_count == n_images
                 else 'unverified_or_mismatched')
    invalid_actions = sum(not is_valid_action(c) for c in all_controls)
    moving_actions = sum(float(c.get('throttle', 0)) > .05 or float(c.get('brake', 0)) > .05
                         for c in all_controls if is_valid_action(c))
    nonzero_steers = sum(abs(float(c.get('steer', 0))) > .05 for c in all_controls if is_valid_action(c))

    box_corrupt = None
    if sample_key is not None and sample_key in boxes:
        try:
            with gzip.open(boxes[sample_key], 'rt', encoding='utf-8') as f:
                json.load(f)
            box_corrupt = False
        except Exception:
            box_corrupt = True

    return {
        'driver': driver,
        'scenario': scenario.name,
        'images': n_images,
        'boxes': len(boxes),
        'metric_info_records': len(infos),
        'control_records': len(records),
        'image_box_id_matches': box_matches,
        'image_telemetry_id_matches': info_matches,
        'image_control_step_matches': action_matches,
        'all_three_exact_id_matches': aligned_count,
        'alignment_hint': alignment,
        'control_duplicated_steps': duplicates,
        'invalid_actions': invalid_actions,
        'action_throttle_or_brake_over_005': moving_actions,
        'action_abs_steer_over_005': nonzero_steers,
        'sample_frame_id': sample_key,
        'sample_info_keys': sorted(sample_info),
        'sample_info_values': {k: sample_info.get(k) for k in
                               ('speed', 'speed_limit', 'location', 'rotation',
                                'target_point', 'target_point_next', 'distance_to_front_vehicle')
                               if k in sample_info},
        'sample_control_from_exact_step': sample_action,
        'sample_box_corrupt': box_corrupt,
        'read_errors': [x for x in (err_info, err_actions) if x],
    }


def probe_metadrive(rgb=False):
    print('\n=== MetaDrive probe (RGB camera) ===' if rgb else '\n=== MetaDrive probe (default environment) ===')
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        if rgb:
            from metadrive.component.sensors.rgb_camera import RGBCamera
        import numpy as np
    except Exception as e:
        print('Not available in this Python environment:', str(e))
        return {'available': False, 'error': str(e)}
    env = None
    try:
        cfg = dict(use_render=False, num_scenarios=1, traffic_density=0.0)
        if rgb:
            cfg.update(image_observation=True,
                       vehicle_config={"image_source": "rgb_camera"},
                       sensors={"rgb_camera": (RGBCamera, 200, 100)},
                       stack_size=1)
        env = MetaDriveEnv(cfg)
        obs, info = env.reset(seed=0)
        print('observation_space =', env.observation_space)
        print('action_space      =', env.action_space)
        print('observation type  =', type(obs).__name__)
        print('observation shape =', getattr(obs, 'shape', None))
        if isinstance(obs, dict):
            print('observation keys  =', list(obs))
            print('image shape       =', getattr(obs.get('image'), 'shape', None))
        else:
            print('observation sample=', np.asarray(obs).reshape(-1)[:12].tolist())
        next_obs, reward, terminated, truncated, step_info = env.step([0.0, 0.0])
        print('step success      =', True)
        print('step info keys    =', sorted(step_info.keys())[:25])
        return {
            'available': True,
            'observation_space': str(env.observation_space),
            'action_space': str(env.action_space),
            'observation_shape': str(getattr(obs, 'shape', None)),
            'observation_keys': list(obs) if isinstance(obs, dict) else None,
            'image_shape': str(getattr(obs.get('image'), 'shape', None)) if isinstance(obs, dict) else None,
            'step_success': True,
        }
    except Exception as e:
        print('MetaDrive probe failed:', type(e).__name__, str(e))
        return {'available': True, 'probe_failed': f'{type(e).__name__}: {e}'}
    finally:
        if env is not None:
            env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pdd_dir', type=Path, help='Folder containing driver_01/, driver_02/, driver_03/')
    p.add_argument('--no-metadrive', action='store_true', help='Skip runtime simulator probe')
    p.add_argument('--rgb', action='store_true', help='Test RGB camera observation (requires offscreen rendering)')
    p.add_argument('--report', type=Path, default=Path('pdd_audit.json'))
    args = p.parse_args()
    root = args.pdd_dir.expanduser().resolve()
    if not root.is_dir():
        p.error(f'Folder does not exist: {root}')

    result = []
    for driver_id in ('driver_01', 'driver_02', 'driver_03'):
        data_dir = root / driver_id / 'data'
        if not data_dir.is_dir():
            print(f'MISSING: {data_dir}')
            continue
        scenarios = sorted(x for x in data_dir.iterdir() if x.is_dir())
        print(f'\n=== {driver_id}: {len(scenarios)} scenarios ===')
        for scene in scenarios:
            row = audit_scenario(driver_id, scene)
            result.append(row)
            n = row['images']
            print(f'{scene.name:38s} imgs={n:4d} boxes={row["image_box_id_matches"]:4d} '
                  f'info={row["image_telemetry_id_matches"]:4d} '
                  f'ctrl={row["image_control_step_matches"]:4d} '
                  f'control_total={row["control_records"]:4d} '
                  f'alignment={row["alignment_hint"]}')

    num_images = sum(r['images'] for r in result)
    matched = sum(r['all_three_exact_id_matches'] for r in result)
    print('\n=== SUMMARY ===')
    print('Scenarios:', len(result), 'Images:', num_images,
          'Images with same numeric ID in all 3 sources:', matched)
    print('WARNING: matching numeric IDs is a necessary clue, NOT proof of time synchronization.')
    print('WARNING: raw CARLA world coordinates cannot be injected as MetaDrive navigation coordinates.')
    print('WARNING: PDD has no native MetaDrive LiDAR vectors.')
    if result:
        row = next((r for r in result if r['images']), result[0])
        print('\nExample:', row['driver'], row['scenario'], 'frame', row['sample_frame_id'])
        print('Telemetry keys:', row['sample_info_keys'])
        print('Telemetry values:', row['sample_info_values'])
        print('Exact-step action:', row['sample_control_from_exact_step'])

    md = {'skipped': True} if args.no_metadrive else probe_metadrive(rgb=args.rgb)
    output = {'pdd_root': str(root), 'scenarios': result,
              'summary': {'scenarios': len(result), 'images': num_images,
                          'exact_three_way_frame_id_matches': matched},
              'metadrive': md}
    args.report.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('\nSaved report:', args.report.resolve())


if __name__ == '__main__':
    main()
