"""Audit CARLA/PDD right-vector and steering sign using raw telemetry.

This script is observational only. It never modifies CSVs, models, or PDD files.
Usage: python agent4_calibration_v21/audit_pdd_v21.py ./PDD --out pdd_calibration_v21.json
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def num(val):
    try:
        x = float(val)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def corr(rows):
    if len(rows) < 25:
        return None
    avg_x = sum(x for x, y in rows) / len(rows)
    avg_y = sum(y for x, y in rows) / len(rows)
    ssx = sum((x - avg_x) ** 2 for x, y in rows)
    ssy = sum((y - avg_y) ** 2 for x, y in rows)
    if ssx < 1e-14 or ssy < 1e-14:
        return None
    return round(sum((x - avg_x) * (y - avg_y) for x, y in rows) / math.sqrt(ssx * ssy), 5)


def pct_sign_agree(rows, cutoff_x, cutoff_y):
    eligible = [(x, y) for x, y in rows if abs(x) >= cutoff_x and abs(y) >= cutoff_y]
    return {'n': len(eligible), 'same_sign_pct': round(100*sum(x*y > 0 for x,y in eligible)/len(eligible), 2) if eligible else None}


def quantiles(vals):
    vals = sorted(x for x in vals if x is not None and math.isfinite(x))
    if not vals:
        return None
    return {k: round(vals[int(p*(len(vals)-1))], 5) for k, p in [('p10', .1), ('median', .5), ('p90', .9)]}


def process_scene(scene, agg):
    info_path = scene / 'metric' / 'metric_info.json'
    ctl_path = scene / 'metric' / 'metrics.json'
    if not info_path.is_file() or not ctl_path.is_file():
        agg['missing_scenes'] += 1
        return
    try:
        infos = json.loads(info_path.read_text(encoding='utf-8-sig'))
        records = json.loads(ctl_path.read_text(encoding='utf-8-sig'))['records']
    except (OSError, ValueError, TypeError, KeyError):
        agg['bad_scenes'] += 1
        return
    actions = {}
    for row in records:
        if isinstance(row, dict):
            c = row.get('control')
            if isinstance(c, dict) and type(c.get('step')) is int:
                actions[c['step']] = c
    timeline = sorted((int(k), v) for k,v in infos.items() if k.isdecimal() and isinstance(v,dict))
    agg['scenes'] += 1
    for index in range(len(timeline)-1):
        s0,i0=timeline[index]
        s1,i1=timeline[index+1]
        if s1 != s0 + 5:
            continue
        rot0,rot1=i0.get('rotation'),i1.get('rotation')
        if not (isinstance(rot0,list) and isinstance(rot1,list) and len(rot0)>=3 and len(rot1)>=3):
            continue
        yaw0,yaw1=num(rot0[2]),num(rot1[2])
        speed=num(i0.get('speed'))
        if yaw0 is None or yaw1 is None or speed is None or speed < 2.0:
            continue
        yaw = (yaw1-yaw0+180)%360-180
        if abs(yaw)>45:
            continue
        controls = [actions.get(t) for t in range(s0,s0+5)]
        if any(c is None for c in controls):
            continue
        steers=[num(c.get('steer')) for c in controls]
        if any(v is None for v in steers):
            continue
        steer = sum(steers)/5
        agg['steer_yaw_pairs'].append((steer, yaw))
        agg['speed_samples_mps'].append(speed)
        target = i0.get('target_point')
        if isinstance(target,(list,tuple)) and len(target)>=2:
            x,y=num(target[0]),num(target[1])
            if x is not None and y is not None:
                agg['wp1_distance_m'].append(math.hypot(x,y))
                agg['wp1_right_raw_m'].append(y)
                agg['steer_vs_wp1_right'].append((steer,y))
        # Future actual displacement: +4 frames / 1 second, projected onto
        # CARLA-reported ego right-vector. Check whether target lateral sign
        # predicts future right displacement. Does not certify route convention.
        if index+4 < len(timeline):
            sn, ine = timeline[index+4]
            if sn - s0 == 20:
                p0,p1=i0.get('location'),ine.get('location')
                right=i0.get('right_vector')
                if (isinstance(p0,list) and isinstance(p1,list) and
                    isinstance(right,list) and len(p0)>=2 and len(p1)>=2 and len(right)>=2):
                    nums=[num(p0[0]),num(p0[1]),num(p1[0]),num(p1[1]),num(right[0]),num(right[1])]
                    if all(v is not None for v in nums) and isinstance(target,(tuple,list)) and len(target)>1:
                        wp_right=num(target[1])
                        if wp_right is not None:
                            actual_right=(nums[2]-nums[0])*nums[4]+(nums[3]-nums[1])*nums[5]
                            if abs(actual_right) < 100:
                                agg['target_right_vs_future_motion'].append((wp_right, actual_right))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdd_root', type=Path)
    parser.add_argument('--out', type=Path, default=Path('pdd_calibration_v21.json'))
    args=parser.parse_args()
    root=args.pdd_root.resolve()
    if not root.is_dir():
        parser.error(f'Missing directory: {root}')
    agg=defaultdict(list)
    agg['scenes']=0; agg['missing_scenes']=0;agg['bad_scenes']=0
    for driver in ('driver_01','driver_02','driver_03'):
        scene_dir=root/driver/'data'
        if not scene_dir.is_dir():
            parser.error(f'Missing {scene_dir}')
        for scene in sorted(scene_dir.iterdir()):
            if scene.is_dir():
                process_scene(scene,agg)
    raw=agg['steer_yaw_pairs']
    result={
        'status':'PDD_WAYPOINT_STEER_DIAGNOSTIC_NOT_COORDINATE_CERTIFICATE',
        'source':str(root),'scenes_seen':agg['scenes'],
        'missing_scenes':agg['missing_scenes'],'bad_scenes':agg['bad_scenes'],
        'steer_vs_yaw':{'n':len(raw),
                         'pearson_corr_steer_vs_next_025s_yaw_deg':corr(raw),
                         'sign_agreement':pct_sign_agree(raw,.03,.2)},
        'waypoint_distances_m':{'wp1':quantiles(agg['wp1_distance_m']),
                                'wp1_right_raw':quantiles(agg['wp1_right_raw_m'])},
        'steer_vs_wp1_right':{'n':len(agg['steer_vs_wp1_right']),
                              'pearson_corr':corr(agg['steer_vs_wp1_right'])},
        'target_right_vs_future_rightward_motion':{
            'n':len(agg['target_right_vs_future_motion']),
            'pearson_corr':corr(agg['target_right_vs_future_motion']),
            'sign_agreement':pct_sign_agree(agg['target_right_vs_future_motion'],1.0,.2),
            'description':'Target lateral coordinate vs rightward displacement over ~1s using CARLA right_vector; lookahead mismatch may weaken correlation.'
        },
        'limits':[
            'CARLA raw yaw is in degrees. CARLA positive yaw/steer conventions differ from MetaDrive; do not infer sign by comparing yaw numbers alone.',
            'A 1-second motion does not necessarily follow a 25m waypoint on curved roads.',
            'Vehicle lane changes, braking and route plans confound raw correlations.',
            'Action labels are from original PDD controls; frame/control synchronization is still a statistical candidate.',
        ]
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Scenes:',result['scenes_seen'])
    print('Steer-vs-yaw corr:',result['steer_vs_yaw']['pearson_corr_steer_vs_next_025s_yaw_deg'])
    print('Target-right-vs-future-right corr:',result['target_right_vs_future_rightward_motion']['pearson_corr'])
    print('Saved:',args.out.resolve())
    print('No files changed; no calibration factors applied.')

if __name__=='__main__':
    main()
