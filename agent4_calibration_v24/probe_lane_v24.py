"""V2.4 shadow-only: compare road-center V2.3 vs lane-center planned route.

Reference action remains [0,0.28]. This does NOT test autonomous driving.
By default probes straight (S), curved (C), and intersection (X) maps.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import joblib

HERE=Path(__file__).resolve().parent
for folder,required in [(HERE.parent/'agent4_pdd_v2','shared_adapter_v2.py'),
                        (HERE.parent/'agent4_calibration_v22','coordinate_patch_v22.py'),
                        (HERE.parent/'agent4_calibration_v23','route_lookahead_v23.py')]:
    if not (folder/required).is_file():
        raise RuntimeError('Missing dependency: '+str(folder/required))
    sys.path.insert(0,str(folder))
from shared_adapter_v2 import FEATURES, metadrive_features
from coordinate_patch_v22 import determine_local_y_to_right_multiplier,correct_features_to_pdd_right,carla_to_metadrive_steer
from route_lookahead_v23 import sample_route_waypoints
from lane_route_v24 import sample_lane_route_waypoints


def statistics(vals):
    if not vals:return None
    a=np.asarray(vals,dtype=float)
    return {'min':float(np.min(a)),'median':float(np.median(a)),
            'p90':float(np.quantile(a,.9)),'max':float(np.max(a)),'count':len(a)}


def prediction(bundle, features):
    x=np.array([[features[k] for k in FEATURES]],dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite input')
    return (float(bundle['models']['steering'].predict(x)[0]),
            float(bundle['models']['longitudinal'].predict(x)[0]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('agent4_v24_lane_probe.json'))
    p.add_argument('--maps',default='S,C,X',help='Comma-separated MetaDrive block maps')
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--wp1',type=float,default=25.0)
    p.add_argument('--wp2',type=float,default=45.0)
    p.add_argument('--max-gap',type=float,default=2.0)
    args=p.parse_args()
    maps=[x.strip() for x in args.maps.split(',') if x.strip()]
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    if not maps or not seeds or min(seeds)<0 or args.steps<1 or not 0<args.wp1<args.wp2<=50 or args.max_gap<=0:
        p.error('Invalid map/seeds/steps/lookahead/gap')
    bundle=joblib.load(args.model)
    if list(bundle.get('features',[]))!=FEATURES or bundle.get('label_mode')!='at_t':
        raise ValueError('Expected the original PDD V2 at_t model with same feature order')
    from metadrive import MetaDriveEnv
    results=[]
    for map_type in maps:
        records,errors,episodes=[],[],[]
        env=None
        try:
            env=MetaDriveEnv({'use_render':False,'map':map_type,'traffic_density':0.0,
                              'num_scenarios':max(seeds)+1,'log_level':40})
            for seed in seeds:
                ns=0; end=None
                try:
                    env.reset(seed=seed)
                    for t in range(args.steps):
                        sign,_=determine_local_y_to_right_multiplier(env.agent)
                        f_raw,_=metadrive_features(env)
                        f=correct_features_to_pdd_right(f_raw,sign)
                        road_wp,road_diag=sample_route_waypoints(env,(args.wp1,args.wp2),sign,args.max_gap)
                        lane_wp,lane_diag=sample_lane_route_waypoints(env,(args.wp1,args.wp2),sign,args.max_gap)
                        f_road=dict(f);f_lane=dict(f)
                        for target,wps in ((f_road,road_wp),(f_lane,lane_wp)):
                            target.update({'wp1_forward_m':wps[0][0], 'wp1_right_m':wps[0][1],
                                           'wp2_forward_m':wps[1][0], 'wp2_right_m':wps[1][1]})
                        usable=road_diag['usable_for_shadow'] and lane_diag['usable_for_shadow']
                        if usable:
                            steer_r,long_r=prediction(bundle,f_road)
                            steer_l,long_l=prediction(bundle,f_lane)
                        else:
                            steer_r=long_r=steer_l=long_l=None
                        records.append({
                            'seed':seed,'step':t,
                            'ego_xy':list(map(float,env.agent.position[:2])),
                            'ego_lane_index':str(getattr(env.agent.lane,'index','unknown')),
                            'native_waypoints':[[f['wp1_forward_m'],f['wp1_right_m']],
                                                [f['wp2_forward_m'],f['wp2_right_m']]],
                            'road_center_waypoints':road_wp,
                            'own_lane_waypoints':lane_wp,
                            'road_route_geometry':road_diag,
                            'own_lane_route_geometry':lane_diag,
                            'usable':usable,
                            'steer_road_carla':steer_r,'steer_lane_carla':steer_l,
                            'steer_lane_metadrive_sign_only':(carla_to_metadrive_steer(steer_l) if usable else None),
                            'longitudinal_road':long_r,'longitudinal_lane':long_l,
                        })
                        ns+=1
                        _,_,term,trunc,info=env.step([0.,0.28])
                        if term or trunc:
                            end={k:bool(info.get(k,False)) for k in ('arrive_dest','out_of_road','crash')}
                            break
                except Exception as e:
                    errors.append({'seed':seed,'step':ns,'error':type(e).__name__+': '+str(e)})
                episodes.append({'seed':seed,'n_samples':ns,'reference_episode_end':end})
        except Exception as e:
            errors.append({'stage':'init','error':type(e).__name__+': '+str(e)})
        finally:
            if env is not None:
                try:env.close()
                except Exception as e:errors.append({'stage':'close','error':str(e)})
        ok=[r for r in records if r['usable']]
        diffs=[abs(r['road_center_waypoints'][0][1]-r['own_lane_waypoints'][0][1]) for r in ok]
        steer_changes=[abs(r['steer_road_carla']-r['steer_lane_carla']) for r in ok]
        summary={
            'n_samples':len(records),'n_usable':len(ok),
            'n_road_geometry_discontinuous':sum(not r['road_route_geometry']['geometry_continuous'] for r in records),
            'n_own_lane_geometry_discontinuous':sum(not r['own_lane_route_geometry']['lane_chain_continuous'] for r in records),
            'n_own_lane_route_incomplete':sum(not r['own_lane_route_geometry']['complete'] for r in records),
            'road_wp1_right_m':statistics([r['road_center_waypoints'][0][1] for r in ok]),
            'lane_wp1_right_m':statistics([r['own_lane_waypoints'][0][1] for r in ok]),
            'road_vs_lane_wp1_right_abs_m':statistics(diffs),
            'steering_prediction_abs_change':statistics(steer_changes),
            'n_road_lane_diff_over_2m':sum(d>2 for d in diffs),
            'n_road_lane_diff_over_3m':sum(d>3 for d in diffs),
        }
        results.append({'map':map_type,'episodes':episodes,'summary':summary,
                        'errors':errors,
                        'examples':records[:min(12,len(records))],
                        'high_lateral_disagreement_examples':sorted(ok,key=lambda r:abs(r['road_center_waypoints'][0][1]-r['own_lane_waypoints'][0][1]),reverse=True)[:5],
                        'invalid_examples':[r for r in records if not r['usable']][:5]})
        print('map %s: %d/%d usable; lane gaps=%d; |road-lane wp1 right| median=%s; errors=%d' % (
            map_type,len(ok),len(records),summary['n_own_lane_geometry_discontinuous'],
            (None if summary['road_vs_lane_wp1_right_abs_m'] is None else round(summary['road_vs_lane_wp1_right_abs_m']['median'],3)),len(errors)))
    report={'status':'V24_LANE_VS_ROAD_CENTER_SHADOW_ONLY','model_action_applied':False,
            'reference_action':[0.0,0.28],'maps':maps,'seeds':seeds,
            'wp_lookahead_m':[args.wp1,args.wp2], 'results':results,
            'warnings':['NOT closed-loop; no actions from the model are applied.',
                        'Nearest continuous lane may be wrong for required turns or lane changes.',
                        'Reference action is straight steering and may end episodes early on curves/intersections.',
                        'PDD targets are variable horizon; waypoint and action equivalence unproven.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',args.out.resolve())

if __name__=='__main__':main()
