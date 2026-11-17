"""V3.14 diagnostic evaluator: independent V3.13 learned steering, full geometry trace.

Never calls IDMPolicy; no steering teacher fallback. Throttle is a fixed P governor,
NOT learned. Uses identical pre-action observation extraction as V3.12B.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import joblib

HERE=Path(__file__).resolve().parent
V313=HERE.parent/"agent4_steering_v313"
if not (V313/"features_v313.py").is_file():
    raise FileNotFoundError(f"Missing V3.13 folder next to V3.14: {V313}")
sys.path.insert(0,str(V313))
from features_v313 import VERSION,steering_prediction
COLLECTOR=HERE.parent/'agent4_idm_dataset_v312b'
if not (COLLECTOR/'collect_v312b.py').is_file():
    raise FileNotFoundError(f'Missing V3.12B collector sibling: {COLLECTOR}. Extract V3.12B folder next to V3.13.')
sys.path.insert(0,str(COLLECTOR))
from collect_v312b import parse_seed_spec,parse_maps,state_before_action,approximate_vehicle_clearance  # noqa: E402

STEP_S=.1


def atom(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)


def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def governor(speed,target,kp=.23,limit=.4):
    if not math.isfinite(speed) or not math.isfinite(target) or target<=0:
        raise ValueError('Bad longitudinal input')
    return max(-limit,min(limit,kp*(target-speed)))


def get_flags(info):
    return {k:bool(info.get(k)) for k in ('arrive_dest','out_of_road','crash','crash_vehicle','crash_object','on_yellow_continuous_line')}


def run_one(env,bundle,m,seed,speed,steps=2500,trace_stride=20):
    env.reset(seed=seed)
    status='step_limit';errors=[];actions=0;checked=0
    max_abs_lateral=0.;max_abs_heading=0.;min_clear=None
    max_speed=0.;speed_sum=0.;near_edge=0;road_marked=0;has_curve=False;last_flags={};trace=[];missing_lookahead_5_10=0
    route_completion=None
    for step in range(steps):
        try:
            pre=state_before_action(env.agent)
            lateral=float(pre['planned_lateral_right_m'])
            heading=float(pre['heading_error_rad'])
            current_speed=float(pre['speed_mps'])
            margin=pre.get('edge_clearance_proxy_m')
            max_abs_lateral=max(max_abs_lateral,abs(lateral))
            max_abs_heading=max(max_abs_heading,abs(heading))
            max_speed=max(max_speed,current_speed);speed_sum+=current_speed
            has_curve|=bool(pre['is_circular_lane'] or abs(pre['curvature_proxy_rad_per_m'])>1e-5)
            if margin is not None:
                min_clear=margin if min_clear is None else min(min_clear,margin)
                near_edge+=int(margin<.35)
            missing_lookahead_5_10+=int(not pre['lookahead_5m_available'] or not pre['lookahead_10m_available'])
            steer=steering_prediction(bundle,pre) # NO IDM, NO TEACHER
            longitude=governor(current_speed,speed)
            result=env.step([steer,longitude])
            if len(result)==5:
                _,_,terminated,truncated,info=result
            elif len(result)==4:
                _,_,terminated,info=result
                truncated=False
            else:raise RuntimeError(f'Unexpected MetaDrive step return length {len(result)}')
            actions+=1
            actual=tuple(float(x) for x in env.agent.current_action)
            if len(actual)!=2 or any(not math.isfinite(x) for x in actual):
                raise RuntimeError('Unreadable applied vehicle action')
            if abs(actual[0]-steer)>5e-4 or abs(actual[1]-longitude)>5e-4:
                raise RuntimeError(f'Action mismatch model={steer:.6f},speed={longitude:.6f} vs applied={actual}')
            checked+=1
            last_flags=get_flags(info)
            road_marked+=int(last_flags['on_yellow_continuous_line'])
            if info.get('route_completion') is not None:
                route_completion=float(info['route_completion'])
            if step%trace_stride==0 or terminated or truncated:
                trace.append(dict(step=step,speed_mps=current_speed,lateral_m=lateral,
                    heading_error_rad=heading,curvature_proxy=pre['curvature_proxy_rad_per_m'],
                    is_circular_lane=bool(pre['is_circular_lane']),predicted_steer=steer,
                    applied_steer=actual[0],applied_longitudinal=actual[1],
                    edge_clearance_proxy_m=margin,
                    planned_lane_id=pre.get('planned_lane_id'),vehicle_lane_id=pre.get('vehicle_lane_id'),
                    planned_lane_class=pre.get('planned_lane_class'),
                    vehicle_lane_class=pre.get('vehicle_lane_class'),
                    distance_to_lane_end_m=pre.get('distance_to_lane_end_m'),
                    selected_next_lane_id=pre.get('selected_next_lane_id'),
                    selected_next_lane_seam_m=pre.get('selected_next_lane_seam_m'),
                    left_side_m=pre.get('left_side_m'),right_side_m=pre.get('right_side_m'),
                    lookahead_5m_available=pre.get('lookahead_5m_available'),
                    lookahead_10m_available=pre.get('lookahead_10m_available'),
                    lookahead_heading_5m_rad=pre.get('lookahead_heading_5m_rad'),
                    lookahead_heading_10m_rad=pre.get('lookahead_heading_10m_rad'),
                    flags=last_flags))
            if terminated or truncated:
                if last_flags.get('arrive_dest'):status='arrive_dest'
                elif last_flags.get('out_of_road'):status='out_of_road'
                elif any(last_flags.get(x) for x in ('crash','crash_vehicle','crash_object')):status='crash'
                elif truncated:status='truncated'
                else:status='other_termination'
                break
        except Exception as e:
            errors.append(f'step {step}: {type(e).__name__}: {e}')
            status='prediction_or_runtime_error'
            break
    arrived_ok=(status=='arrive_dest' and checked==actions and not errors)
    quality_pass=(arrived_ok and max_abs_lateral<=.6 and max_abs_heading<=.12 and
                  min_clear is not None and min_clear>=.3 and near_edge==0 and road_marked==0)
    return dict(map=m,seed=int(seed),speed_mps=float(speed),profile='learned_v313',
        status=status,confirmed_success=arrived_ok,quality_pass=quality_pass,
        n_applied_steps=actions,n_verified_prediction_steps=checked,model_predictions_valid=checked==actions and not errors and actions>0,
        teacher_fallback_calls=0,idm_policy_used=False,learned_steering=True,learned_longitudinal=False,
        elapsed_s=actions*STEP_S,mean_speed_mps=speed_sum/max(actions,1),max_speed_mps=max_speed,
        max_abs_planned_lateral_m=max_abs_lateral,max_abs_heading_error_rad=max_abs_heading,
        min_edge_clearance_proxy_m=min_clear,near_edge_warning_steps=near_edge,
        road_marked_steps=road_marked,
        has_circular_lane=has_curve,missing_lookahead_5_10_steps=missing_lookahead_5_10,
        route_completion=route_completion,termination_flags=last_flags,
        errors=errors,trace=trace)


def aggregate(episodes):
    grouped={}
    for e in episodes:grouped.setdefault((e['map'],e['speed_mps']),[]).append(e)
    return [dict(map=m,speed_mps=v,total=len(es),arrive_dest=sum(x['confirmed_success'] for x in es),
            quality_accepted=sum(x['quality_pass'] for x in es),
            max_lateral_m=max(x['max_abs_planned_lateral_m'] for x in es),
            failures=[dict(seed=x['seed'],status=x['status'],errors=x['errors'][:1])
                      for x in es if not x['confirmed_success']])
            for (m,v),es in sorted(grouped.items())]


def output_csv(report,path):
    cols=('map','seed','speed_mps','status','confirmed_success','n_applied_steps',
          'mean_speed_mps','max_abs_planned_lateral_m','min_edge_clearance_proxy_m',
          'near_edge_warning_steps','road_marked_steps','quality_pass',
          'missing_lookahead_5_10_steps','model_predictions_valid','teacher_fallback_calls')
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for e in report['episodes']:w.writerow({k:e.get(k) for k in cols})


def run(args):
    maps=parse_maps(args.maps)
    seeds=parse_seed_spec(args.seeds)
    if not seeds:raise ValueError('At least one evaluation seed required')
    speeds=[]
    for part in args.speeds.split(','):
        v=float(part.strip())
        if not math.isfinite(v) or not 1<=v<=6:raise ValueError('Speed must be 1..6m/s')
        speeds.append(v)
    if len(speeds)!=len(set(speeds)):raise ValueError('Duplicate speeds')
    if args.steps<100 or args.trace_stride<1:raise ValueError('Invalid episode limits')
    bundle=joblib.load(args.model) # Only load a trusted model file from your own training.
    if bundle.get('version')!=VERSION or bundle.get('teacher_fallback') is not False:
        raise ValueError('Not a signed-off V3.13 no-fallback steering model')
    config=dict(maps=maps,seeds=seeds,speeds_mps=speeds,steps=args.steps,
                model_sha256=fingerprint(args.model),trace_stride=args.trace_stride,
                throttle_type='rule_based_P_governor',traffic_density=0.0)
    out=Path(args.out)
    if out.exists() and args.resume:
        report=json.loads(out.read_text(encoding='utf8'))
        if report.get('version')!=VERSION or report.get('config')!=config:
            raise ValueError('Model/config changed since previous report; do not resume')
    else:
        if out.exists():raise FileExistsError(f'{out} already exists: use --resume or new output path')
        report=dict(version=VERSION,config=config,episodes=[],groups=[],
            warnings=['V3.14: extra full-step lane identity/geometry trace, no controller changes.',
                      'True closed-loop MODEL steering only, no teacher/IDM call.',
                      'Throttle is hand-coded, not learned.',
                      'Uses privileged MetaDrive lane geometry, zero traffic, simulator only.',
                      'Metrics from test-seed evaluation are not a retraining signal.'])
    seen={(e['map'],e['seed'],e['speed_mps']) for e in report['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    for m in maps:
        todo=[(seed,speed) for seed in seeds for speed in speeds if (m,seed,speed) not in seen]
        if not todo:continue
        env=MetaDriveEnv(dict(map=m,use_render=False,num_scenarios=max(seeds)+1,
             traffic_density=0.0,horizon=max(1000,args.steps+10),log_level=40,
             physics_world_step_size=.02,decision_repeat=5))
        try:
            for seed,speed in todo:
                ep=run_one(env,bundle,m,seed,speed,args.steps,args.trace_stride)
                report['episodes'].append(ep);report['groups']=aggregate(report['episodes'])
                atom(out,report);output_csv(report,out.with_suffix('.csv'))
                print(f'{m} seed={seed} speed={speed:g}m/s status={ep["status"]} '
                      f'confirmed={int(ep["confirmed_success"])} lateral={ep["max_abs_planned_lateral_m"]:.3f}m',flush=True)
        finally:env.close()
    print('JSON:',out.resolve());return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--maps',default='C,X')
    p.add_argument('--seeds',default='190-193')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--trace-stride',type=int,default=1)
    p.add_argument('--out',type=Path,default=Path('agent4_v313_closed_loop.json'))
    p.add_argument('--resume',action='store_true')
    return run(p.parse_args(argv))

if __name__=='__main__':main()
