"""PDD -> MetaDrive bridge diagnostics (NOT a validated training converter).

Windows example:
    python pdd_metadrive_bridge_check.py "D:\\Học\\Tri tue nhan to nang cao\\lab\\dataset\\PDD"

Outputs in --out (default: bridge_output):
  - bridge_report.json          Semantic check for PDD and MetaDrive
  - pdd_candidate_manifest.csv  Candidate (NOT time-verified) image/telemetry/action matches
  - pdd_sample.png              First PDD camera image, resized to 200x100
  - metadrive_sample.png        RGB camera image rendered by MetaDrive

Requires: Python 3.10, numpy, cv2, metadrive-simulator (installed in same Python).
Never train using the candidate manifest until recording timing is verified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def finite_number(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def vec2(x):
    if not isinstance(x, (list, tuple)) or len(x) < 2:
        return None
    a, b = finite_number(x[0]), finite_number(x[1])
    return [a, b] if a is not None and b is not None else None


def shorten(x, maxlen=600):
    s = json.dumps(x, ensure_ascii=False, default=str)
    return s[:maxlen] + (' ...' if len(s) > maxlen else '')


def percentile(values, pct):
    if not values:
        return None
    v = sorted(values)
    return round(v[round((len(v) - 1) * pct)], 5)


def read_pdd(root, outdir):
    """Create candidate rows by frame order vs ordered telemetry steps, not by frame ID.
    Only exact control.step matches allowed. No implicit fill with 0.
    """
    rows = []
    summaries = []
    examples = []
    all_speeds, all_steers, all_a2, all_forward, all_side = [], [], [], [], []
    missing_counts = Counter()
    first_image = None
    for driver in ('driver_01', 'driver_02', 'driver_03'):
        root_data = root / driver / 'data'
        if not root_data.is_dir():
            missing_counts[f'{driver}_directory_missing'] += 1
            continue
        for scene in sorted(p for p in root_data.iterdir() if p.is_dir()):
            src = scene / 'metric'
            info_file = src / 'metric_info.json'
            control_file = src / 'metrics.json'
            if not info_file.is_file() or not control_file.is_file():
                missing_counts['scenario_metric_file_missing'] += 1
                continue
            try:
                infos = json.loads(info_file.read_text(encoding='utf-8'))
                controls = json.loads(control_file.read_text(encoding='utf-8'))['records']
            except (ValueError, KeyError, OSError, TypeError):
                missing_counts['scenario_json_unreadable'] += 1
                continue
            if not isinstance(infos, dict) or not isinstance(controls, list):
                missing_counts['scenario_schema_invalid'] += 1
                continue
            frames = sorted((p for p in (scene / 'images').glob('*.jpg') if p.stem.isdecimal()),
                            key=lambda x: int(x.stem))
            steps = sorted(int(k) for k in infos if str(k).isdecimal())
            control_by_step = {}
            for rec in controls:
                c = rec.get('control', {}) if isinstance(rec, dict) else {}
                if isinstance(c, dict) and type(c.get('step')) is int:
                    control_by_step[c['step']] = c
            scene_count = 0
            for image_path, step in zip(frames, steps):
                info = infos.get(str(step), {})
                ctrl = control_by_step.get(step)
                if not isinstance(info, dict) or not ctrl:
                    missing_counts['missing_telemetry_or_control'] += 1
                    continue
                speed = finite_number(info.get('speed'))
                speed_lim = finite_number(info.get('speed_limit'))
                target1 = vec2(info.get('target_point'))
                target2 = vec2(info.get('target_point_next'))
                dist_front = finite_number(info.get('distance_to_front_vehicle'))
                steer = finite_number(ctrl.get('steer'))
                throttle = finite_number(ctrl.get('throttle'))
                brake = finite_number(ctrl.get('brake'))
                if (speed is None or target1 is None or target2 is None or steer is None
                        or throttle is None or brake is None):
                    missing_counts['invalid_mandatory_fields'] += 1
                    continue
                if not (-1 <= steer <= 1 and 0 <= throttle <= 1 and 0 <= brake <= 1):
                    missing_counts['action_out_of_range'] += 1
                    continue
                a2 = max(-1., min(1., throttle - brake))  # only a numerical candidate
                row = {
                    'driver': driver, 'scenario': scene.name, 'frame_id': int(image_path.stem),
                    'image_path': str(image_path), 'telemetry_step': step,
                    'speed_mps': speed, 'speed_limit_mps': speed_lim,
                    'target1_forward_candidate_m': target1[0],
                    'target1_lateral_candidate_m': target1[1],
                    'target2_forward_candidate_m': target2[0],
                    'target2_lateral_candidate_m': target2[1],
                    'front_dist_m_minus1_if_missing': dist_front,
                    'steer_carla': steer, 'throttle_carla': throttle, 'brake_carla': brake,
                    'metadrive_a2_candidate': a2,
                    'alignment_verified': False,
                }
                rows.append(row)
                scene_count += 1
                all_speeds.append(speed)
                all_steers.append(steer)
                all_a2.append(a2)
                all_forward.append(target1[0])
                all_side.append(target1[1])
                if len(examples) < 4:
                    examples.append({
                        'driver': driver, 'scenario': scene.name,
                        'frame_id': int(image_path.stem), 'telemetry_step': step,
                        'target_point': target1,
                        'lane_info_preview': shorten(info.get('lane_info')),
                        'other_vehicles_preview': shorten(info.get('other_vehicles')),
                        'boxes_path': str(scene / 'boxes' / (image_path.stem + '.json.gz')),
                    })
                if first_image is None:
                    first_image = image_path
            summaries.append({
                'driver': driver, 'scenario': scene.name,
                'images': len(frames), 'telemetry': len(steps), 'controls': len(controls),
                'candidate_pairs': scene_count,
                'first_telemetry_step': steps[0] if steps else None,
                'last_telemetry_step': steps[-1] if steps else None,
                'telemetry_gaps': dict(Counter(b-a for a,b in zip(steps, steps[1:]))),
            })
    fields = [
        'driver','scenario','frame_id','image_path','telemetry_step','speed_mps',
        'speed_limit_mps','target1_forward_candidate_m','target1_lateral_candidate_m',
        'target2_forward_candidate_m','target2_lateral_candidate_m',
        'front_dist_m_minus1_if_missing', 'steer_carla','throttle_carla',
        'brake_carla','metadrive_a2_candidate','alignment_verified'
    ]
    with (outdir / 'pdd_candidate_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    image_report = {'created': False}
    if first_image is not None:
        try:
            import cv2
            import numpy as np
            # Unicode paths on Windows: cv2.imread may fail; use imdecode instead.
            image = cv2.imdecode(np.fromfile(str(first_image), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, (200, 100), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.png', image)
                if not ok:
                    raise RuntimeError('Could not encode PDD PNG')
                encoded.tofile(str(outdir / 'pdd_sample.png'))
                image_report = {'created': True, 'source': str(first_image),
                                'saved_path': str(outdir / 'pdd_sample.png')}
            else:
                image_report['error'] = 'cv2.imread returned None'
        except Exception as ex:
            image_report['error'] = f'{type(ex).__name__}: {ex}'
    return {
        'scenarios_found': len(summaries), 'candidates': len(rows), 'missing': dict(missing_counts),
        'speed_mps_p10_p50_p90': [percentile(all_speeds,p) for p in (.1,.5,.9)],
        'steer_p10_p50_p90': [percentile(all_steers,p) for p in (.1,.5,.9)],
        'a2_candidate_p10_p50_p90': [percentile(all_a2,p) for p in (.1,.5,.9)],
        'target1_forward_p10_p50_p90': [percentile(all_forward,p) for p in (.1,.5,.9)],
        'target1_lateral_p10_p50_p90': [percentile(all_side,p) for p in (.1,.5,.9)],
        'sample_record_previews': examples,
        'scenario_summary': summaries,
        'image': image_report,
        'critical_note': 'Frame-index/order mapping is candidate-only. Do NOT train until original collection timing is validated.',
    }


def probe_metadrive(outdir, steps=20):
    try:
        import numpy as np
        import cv2
        from metadrive import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
    except Exception as ex:
        return {'success': False, 'error': f'{type(ex).__name__}: {ex}'}
    env = None
    try:
        cfg = dict(
            use_render=False, num_scenarios=1, traffic_density=0.0,
            image_observation=True,
            vehicle_config={'image_source': 'rgb_camera'},
            sensors={'rgb_camera': (RGBCamera, 200, 100)},
            stack_size=1,
        )
        env = MetaDriveEnv(cfg)
        obs, info = env.reset(seed=0)
        if not isinstance(obs, dict):
            raise RuntimeError(f'Expected dict RGB observation, got {type(obs).__name__}')
        img = np.asarray(obs['image'])
        if img.shape != (100,200,3,1):
            raise RuntimeError(f'Unexpected camera shape {img.shape}; inspect version/config')
        img = img[...,0]
        if float(img.max()) <= 1.0 + 1e-6:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
        # Camera RGB -> PNG/BGR. Do not flip until visually checked.
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if ok:
            encoded.tofile(str(outdir / 'metadrive_sample.png'))
        saved = bool(ok)
        agent = env.agent
        nav = getattr(agent, 'navigation', None)
        lane = getattr(agent, 'lane', None)
        lane_local = None
        if lane is not None:
            try:
                lane_local = [float(v) for v in lane.local_coordinates(agent.position)]
            except Exception:
                pass
        nav_info = None
        if nav is not None:
            try:
                nav_info = np.asarray(nav.get_navi_info()).reshape(-1).tolist()
            except Exception:
                pass
        pre = {
            'rgb_shape': list(obs['image'].shape),
            'rgb_min_max': [float(np.min(obs['image'])), float(np.max(obs['image']))],
            'state_length': int(np.asarray(obs['state']).size),
            'state_19_normalized_values': np.asarray(obs['state']).reshape(-1).tolist(),
            'speed_kmh': float(agent.speed_km_h),
            'speed_mps': float(agent.speed_km_h) / 3.6,
            'heading_theta_radians': float(agent.heading_theta),
            'position_xy': np.asarray(agent.position).reshape(-1)[:2].tolist(),
            'lane_local_longitudinal_lateral': lane_local,
            'navigation_info_normalized': nav_info,
            'image_created': bool(saved),
        }
        last_info = {}
        for _ in range(steps):
            obs, reward, term, trunc, last_info = env.step([0.0, 0.45])
            if term or trunc:
                break
        post = {
            'speed_kmh': float(agent.speed_km_h),
            'speed_mps': float(agent.speed_km_h)/3.6,
            'route_completion': last_info.get('route_completion'),
            'arrive_dest': last_info.get('arrive_dest'),
            'crash': last_info.get('crash'),
        }
        # Approximate steering sign in MetaDrive; direction of CARLA labels
        # must still be confirmed independently from successive PDD telemetry.
        steering_tests = []
        for steer_cmd in (0.35, -0.35):
            env.reset(seed=0)
            agent = env.agent
            for _ in range(35):
                _, _, term, trunc, _ = env.step([0.0, 0.6])
                if term or trunc:
                    break
            yaw_before = float(agent.heading_theta)
            for _ in range(12):
                _, _, term, trunc, _ = env.step([steer_cmd, 0.35])
                if term or trunc:
                    break
            delta = math.atan2(math.sin(float(agent.heading_theta)-yaw_before),
                               math.cos(float(agent.heading_theta)-yaw_before))
            steering_tests.append({
                'steer_command': steer_cmd,
                'yaw_change_radians': delta,
                'final_speed_mps': float(agent.speed_km_h)/3.6,
            })
        return {'success': True, 'initial': pre, 'after_acceleration': post,
                'steering_tests': steering_tests,
                'action_space': str(env.action_space),
                'notes': 'Steering direction measured in MetaDrive. CARLA label sign and force calibration remain independent.'}
    except Exception as ex:
        return {'success': False, 'error': f'{type(ex).__name__}: {ex}'}
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdd_root', type=Path)
    parser.add_argument('--out', type=Path, default=Path('bridge_output'))
    parser.add_argument('--no-metadrive', action='store_true')
    args = parser.parse_args()
    root = args.pdd_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'PDD root not found: {root}')
    outdir = args.out.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print('Inspect PDD raw files...')
    pdd = read_pdd(root, outdir)
    print(f"PDD: {pdd['scenarios_found']} scenarios, {pdd['candidates']} candidate samples")
    print('Probe MetaDrive...')
    md = {'skipped': True} if args.no_metadrive else probe_metadrive(outdir)
    print('MetaDrive success:', md.get('success'))
    report = {
        'status': 'DIAGNOSTICS_ONLY_NOT_MODEL_COMPATIBILITY_PROOF',
        'pdd': pdd, 'metadrive': md,
        'gates_to_train': [
            'Validate image/telemetry/control timestamps or original data collector',
            'Verify speed units, waypoint coordinate frame/handedness, camera pose',
            'Build same named physical features from PDD and MetaDrive (not native 259/19 vector)',
            'Calibrate steer sign and gas/brake response in MetaDrive',
            'Train a small behavioral cloning baseline and evaluate closed-loop, unseen seeds',
        ],
    }
    path = outdir / 'bridge_report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('Saved:', path)
    print('Saved:', outdir / 'pdd_candidate_manifest.csv')
    if pdd['image']['created']:
        print('Saved:', outdir / 'pdd_sample.png')
    if md.get('success'):
        print('Saved:', outdir / 'metadrive_sample.png')
    print('CAUTION: The CSV contains *candidate* alignments, NOT verified training labels.')

if __name__ == '__main__':
    main()
