"""Compare native MetaDrive V2.2 waypoints and route-aware ~25/45m candidates.

Uses fixed reference action [0, 0.28]. NEVER applies model output to env.step.
Run from folder containing agent4_pdd_v2 and agent4_calibration_v22.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'agent4_pdd_v2'
V22 = HERE.parent / 'agent4_calibration_v22'
for required in [V2 / 'shared_adapter_v2.py', V22 / 'coordinate_patch_v22.py']:
    if not required.is_file():
        raise RuntimeError('Required earlier package not found: %s' % required)
sys.path.insert(0,str(V2))
sys.path.insert(0,str(V22))

from shared_adapter_v2 import FEATURES, metadrive_features   # noqa: E402
from coordinate_patch_v22 import (determine_local_y_to_right_multiplier,
    correct_features_to_pdd_right, carla_to_metadrive_steer)  # noqa: E402
from route_lookahead_v23 import sample_route_waypoints  # noqa: E402


def describe(arr):
    if len(arr)==0:
        return None
    a = np.asarray(arr,dtype=float)
    return {'min':float(a.min()),'median':float(np.median(a)), 'p90':float(np.quantile(a,.9)),
            'max':float(a.max()),'n':len(a)}


def predict(bundle, f):
    x = np.array([[f[k] for k in FEATURES]],dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('non-finite model input')
    return {
        'steer_carla':float(bundle['models']['steering'].predict(x)[0]),
        'longitudinal_candidate':float(bundle['models']['longitudinal'].predict(x)[0])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('agent4_v23_route_probe.json'))
    p.add_argument('--seeds',default='0,1,2,3')
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--map',default='S', help='MetaDrive procedural map string (start with S)')
    p.add_argument('--traffic-density',type=float,default=0.0)
    p.add_argument('--wp1',type=float,default=25.0)
    p.add_argument('--wp2',type=float,default=45.0)
    args=p.parse_args()
    seeds=[int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if not seeds or min(seeds)<0 or args.steps<1 or not 0<=args.traffic_density<=1:
        p.error('Invalid seeds, steps or traffic density')
    if not 0 < args.wp1 < args.wp2 <= 50:
        p.error('Lookaheads must satisfy 0 < wp1 < wp2 <= 50m')
    bundle=joblib.load(args.model)
    if list(bundle.get('features',[]))!=FEATURES:
        raise ValueError('Model trained on different feature order')
    if bundle.get('label_mode')!='at_t':
        raise ValueError('This shadow script uses at_t pilot model only')
    from metadrive import MetaDriveEnv
    env = MetaDriveEnv({'use_render':False,'map':args.map,
                        'traffic_density':args.traffic_density,
                        'num_scenarios':max(seeds)+1, 'log_level':40})
    samples=[]; episodes=[]; errors=[]
    try:
        for seed in seeds:
            try:
                env.reset(seed=seed)
                count=0; reason=None
                for t in range(args.steps):
                    sign, axis_detail=determine_local_y_to_right_multiplier(env.agent)
                    base_raw, _ = metadrive_features(env)
                    corrected = correct_features_to_pdd_right(base_raw,sign)
                    road_wp, road_diag=sample_route_waypoints(env,(args.wp1,args.wp2),sign)
                    revised = dict(corrected)
                    revised.update({'wp1_forward_m':road_wp[0][0], 'wp1_right_m':road_wp[0][1],
                                    'wp2_forward_m':road_wp[1][0], 'wp2_right_m':road_wp[1][1]})
                    prev_pred=predict(bundle,corrected)
                    route_pred=predict(bundle,revised)
                    # Shadow only; this never becomes the applied action.
                    record={
                        'seed':seed,'step':t,'speed_mps':float(corrected['speed_mps']),
                        'native_wp1':[float(corrected['wp1_forward_m']),float(corrected['wp1_right_m'])],
                        'native_wp2':[float(corrected['wp2_forward_m']),float(corrected['wp2_right_m'])],
                        'route_wp1':list(road_wp[0]),'route_wp2':list(road_wp[1]),
                        'route_geometry':road_diag,
                        'steer_carla_before':prev_pred['steer_carla'],
                        'steer_carla_after':route_pred['steer_carla'],
                        'steer_metadrive_sign_only_candidate':carla_to_metadrive_steer(route_pred['steer_carla']),
                        'longitudinal_before':prev_pred['longitudinal_candidate'],
                        'longitudinal_after':route_pred['longitudinal_candidate'],
                        'raw_lateral_to_right_sign':sign,
                    }
                    samples.append(record)
                    count+=1
                    _,_,terminated,truncated,info=env.step([0.0,0.28])
                    if terminated or truncated:
                        reason={k:bool(info.get(k,False)) for k in ('arrive_dest','out_of_road','crash')}
                        break
                episodes.append({'seed':seed, 'samples':count,'reference_episode_end':reason})
            except Exception as exc:
                errors.append({'seed':seed,'after_samples':sum(r['seed']==seed for r in samples),
                               'error':type(exc).__name__+': '+str(exc)})
    finally:
        env.close()
    good=[x for x in samples if x['route_geometry']['usable_for_shadow']]
    summary={
        'n_samples':len(samples), 'n_usable':len(good),
        'n_incomplete_route':sum(not x['route_geometry']['complete'] for x in samples),
        'n_geometry_discontinuous':sum(not x['route_geometry']['geometry_continuous'] for x in samples),
        'native_wp1_forward_m':describe([x['native_wp1'][0] for x in samples]),
        'route_wp1_forward_m':describe([x['route_wp1'][0] for x in good]),
        'native_wp2_forward_m':describe([x['native_wp2'][0] for x in samples]),
        'route_wp2_forward_m':describe([x['route_wp2'][0] for x in good]),
        'native_wp1_right_m':describe([x['native_wp1'][1] for x in samples]),
        'route_wp1_right_m':describe([x['route_wp1'][1] for x in good]),
        'steer_metadrive_sign_only_candidate':describe([x['steer_metadrive_sign_only_candidate'] for x in good]),
        'steer_change_abs':describe([abs(x['steer_carla_after']-x['steer_carla_before']) for x in good]),
    }
    warnings=['Route-aware lookahead is EXPERIMENTAL; PDD targets use their own variable horizon.',
              'This is privileged simulator navigation, not a camera-based agent.',
              'No model actions were sent to MetaDrive. Steering gain and longitudinal dynamics not calibrated.']
    if summary['n_incomplete_route']:
        warnings.append('Some lookahead targets extend beyond remaining route: do not use those predictions')
    if summary['n_geometry_discontinuous']:
        warnings.append('Route lane-centre discontinuities detected: route-following geometry may be unsafe')
    if errors:
        warnings.append('One or more seeds failed; inspect errors')
    report={'status':'V23_ROUTE_LOOKAHEAD_SHADOW_NOT_CLOSED_LOOP',
            'model_action_applied':False, 'reference_action':[0.0,0.28],
            'map':args.map,'traffic_density':args.traffic_density,
            'lookahead_m':[args.wp1,args.wp2],
            'episodes':episodes,'summary':summary,'errors':errors,
            'examples':samples[:min(20,len(samples))],
            'invalid_examples':[r for r in samples if not r['route_geometry']['usable_for_shadow']][:10],
            'warnings':warnings}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Native vs route-aware waypoint probe:',len(samples),'samples; usable:',len(good))
    print('Incomplete route:',summary['n_incomplete_route'], 'geometry discontinuities:',summary['n_geometry_discontinuous'])
    print('Errors:',errors[:3]);print('Model action applied: NO')
    print('Saved:',args.out.resolve())

if __name__=='__main__':
    main()
