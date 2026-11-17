"""Agent4 V3.0: baseline vs bounded residual steering on the SAME MetaDrive seeds.

No retraining, no IDM, no learner-generated accelerator/brake. In simulation
only. Output includes road-line termination flags and per-step tail diagnostics.
Requires sibling V2.7 and V2.9 directories plus their dependencies.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder, file in [('agent4_calibration_v27','guarded_pilot_v27.py'),
                     ('agent4_diagnostic_v29','diagnose_road_flags_v29.py')]:
    path = ROOT / folder / file
    if not path.is_file():
        raise RuntimeError(f'Missing dependency {path}; extract V3.0 beside V2.7 and V2.9')
    sys.path.insert(0, str(path.parent))

from guarded_pilot_v27 import (  # noqa: E402
    FEATURES, feature_for_model, predict_steering, governed_speed_action,
    lane_offset, choose_lane_chain, collect_forward_segments)
from diagnose_road_flags_v29 import (  # noqa: E402
    compact_lane_state, physical_road_state, safe_flags)
from residual_controller_v30 import (  # noqa: E402
    rightward_lateral_error, wrap_angle, finite, steering_action,
    estimated_edge_clearance_m)


def planned_geometry(agent, max_gap):
    """Signed right-offset and heading error relative to the selected route lane."""
    chain, gaps = choose_lane_chain(
        collect_forward_segments(agent.navigation, agent), agent.position,
        max_gap_m=max_gap)
    if gaps or not chain:
        raise ValueError(f'route lane chain gaps: {gaps}')
    lane = chain[0]['lane']
    lateral_right, longitude, lateral_axis_sign = rightward_lateral_error(lane, agent.position)
    length = finite('lane.length', lane.length)
    s0 = max(0.0, min(length - 1e-4, longitude - 0.2))
    s1 = max(s0 + 1e-4, min(length, longitude + 0.2))
    p0, p1 = lane.position(s0,0),lane.position(s1,0)
    tangent = math.atan2(float(p1[1])-float(p0[1]),float(p1[0])-float(p0[0]))
    heading = finite('vehicle heading', agent.heading_theta)
    err = wrap_angle(heading - tangent)
    return {
        'lane':repr(chain[0].get('nodes')),
        'lane_index':chain[0].get('lane_index'),
        'current_vehicle_lane':repr(getattr(agent,'lane_index',None)),
        'lateral_right_m':lateral_right,
        'heading_error_rad':err,
        'tangent_rad':tangent,
        'local_lane_lateral_to_right_sign':lateral_axis_sign,
    }


def status_from_info(info):
    if bool(info.get('arrive_dest',False)):
        return 'arrive_dest'
    if bool(info.get('out_of_road',False)):
        return 'out_of_road'
    if bool(info.get('crash',False)):
        return 'crash'
    return 'terminated_or_truncated'


def pilot_episode(env, seed, model, mode, options):
    errors=[]; sample=[]; tail=deque(maxlen=options.tail_steps)
    status='step_limit'; end_info=None; prev=0.0; travelled=0.0
    n=0; residual_count=0; sat_count=0; slew_count=0; adaptive_count=0
    max_abs_offset=0.0; max_heading=0.0; minimum_estimated_margin=None
    road_flag_alerts=0; near_line_alerts=0; last_road=None
    try:
        env.reset(seed=seed)
        for t in range(options.steps):
            car=env.agent
            before_road=physical_road_state(env,car)
            if before_road['predicted_out_of_road']:
                status='preexisting_out_of_road_guard'; last_road=before_road; break
            offset,width=lane_offset(car)
            if offset is not None and abs(offset)>options.guard_lane_offset_m:
                status='lane_departure_guard'; break
            try:
                features,route=feature_for_model(env,options.max_gap)
                geom=planned_geometry(car,options.max_gap)
                heading_err=geom['heading_error_rad']
                if abs(heading_err)>options.guard_heading_rad:
                    status='heading_departure_guard'; break
                clearance=estimated_edge_clearance_m(car,heading_err)
                margin=clearance['minimum']
                if options.guard_road_margin_m>0 and margin < options.guard_road_margin_m:
                    status='road_margin_guard'; last_road=before_road; break
                carla_steer=predict_steering(model,features)
                applied=steering_action(
                    carla_steer,mode,geom['lateral_right_m'],heading_err,
                    options.kp_lateral,options.kp_heading,options.residual_limit,
                    options.steer_gain,options.steer_limit,prev,options.max_steer_delta)
                longitudinal,target_speed=governed_speed_action(
                    features['speed_mps'],options.speed_limit_mps,route['remaining_m'])
            except Exception as exc:
                status='observation_or_model_guard'
                errors.append(f'step {t}: {type(exc).__name__}: {exc}')
                break
            action=[float(applied['steer']),float(longitudinal)]
            before_xy=np.asarray(car.position[:2],dtype=float)
            try:
                _,reward,terminated,truncated,info=env.step(action)
            except Exception as exc:
                status='env_step_error'
                errors.append(f'step {t}: {type(exc).__name__}: {exc}')
                break
            after_xy=np.asarray(env.agent.position[:2],dtype=float)
            travelled+=float(np.linalg.norm(after_xy-before_xy))
            after_road=physical_road_state(env,env.agent)
            last_road=after_road
            n+=1
            prev=float(applied['steer'])
            adaptive_count += int(route['adapted'])
            sat_count += int(applied['steering_saturation'])
            slew_count += int(applied['slew_limited'])
            residual_count += int(abs(applied['correction']['bounded'])>1e-8)
            max_abs_offset=max(max_abs_offset,abs(geom['lateral_right_m']))
            max_heading=max(max_heading,abs(heading_err))
            minimum_estimated_margin=(margin if minimum_estimated_margin is None
                                      else min(minimum_estimated_margin,margin))
            near_line=margin < options.warn_road_margin_m
            near_line_alerts += int(near_line)
            road_flag_alerts+=int(bool(after_road['active_reasons']))
            event={
                'step':t,'mode':mode,'speed_mps_before':float(features['speed_mps']),
                'target_speed_mps':float(target_speed),
                'model_steer_carla':float(carla_steer),
                'model_steer_md':float(applied['base']),
                'residual_steer':float(applied['correction']['bounded']),
                'residual_components':{
                    'lateral':float(applied['correction']['lateral_term']),
                    'heading':float(applied['correction']['heading_term'])},
                'action_applied':action,
                'planned_lateral_right_m':float(geom['lateral_right_m']),
                'heading_error_rad':float(heading_err),
                'lane_index':geom['current_vehicle_lane'],
                'lane_class':type(car.lane).__name__ if car.lane is not None else None,
                'road_margin_approx_m':float(margin),
                'road_flags_after':after_road['flags'],
                'road_causes_after':after_road['active_reasons'],
                'route_completion':float(info.get('route_completion', 0.0)),
                'remaining_route_m':float(route['remaining_m']),
                'lookahead_m':route['effective_lookahead_m'],
                'adapted':bool(route['adapted']),
                'terminated':bool(terminated), 'truncated':bool(truncated),
                'xy_after':after_xy.tolist(),
            }
            tail.append(event)
            if t % options.trace_stride == 0 or terminated or truncated or near_line:
                sample.append(event)
            if terminated or truncated:
                end_info={'terminated':bool(terminated),'truncated':bool(truncated),
                          **safe_flags(info)}
                status=status_from_info(info)
                break
    except Exception as exc:
        status='episode_error'
        errors.append(f'{type(exc).__name__}: {exc}')
    # Sample at requested stride and always keep detailed final window.
    trace_by_step={e['step']:e for e in sample}
    trace_by_step.update({e['step']:e for e in tail})
    return {
        'seed':seed,'mode':mode,'status':status,'n_applied_steps':n,
        'travelled_approx_m':travelled,
        'max_abs_planned_lateral_right_m':max_abs_offset,
        'max_abs_heading_error_rad':max_heading,
        'min_estimated_edge_clearance_m':minimum_estimated_margin,
        'correction_active_steps':residual_count,
        'steering_saturation_steps':sat_count,'steering_slew_limited_steps':slew_count,
        'adaptive_lookahead_steps':adaptive_count,
        'near_edge_warning_steps':near_line_alerts,
        'road_predicate_active_after_steps':road_flag_alerts,
        'terminal_road_state':last_road if status in {'out_of_road','road_margin_guard'} else None,
        'end_info':end_info,'errors':errors,
        'trace':sorted(trace_by_step.values(),key=lambda e:e['step'])}


def summarize_comparison(episodes_by_mode):
    if 'baseline' not in episodes_by_mode or 'residual' not in episodes_by_mode:
        return None
    b,r=episodes_by_mode['baseline'],episodes_by_mode['residual']
    return {
        'baseline_status':b['status'],'residual_status':r['status'],
        'baseline_distance_m':b['travelled_approx_m'],
        'residual_distance_m':r['travelled_approx_m'],
        'distance_delta_m':r['travelled_approx_m']-b['travelled_approx_m'],
        'baseline_route_completion':(b['end_info'] or {}).get('route_completion'),
        'residual_route_completion':(r['end_info'] or {}).get('route_completion'),
        'model_fixed_between_modes':True,
        'note':'Comparison is experimental and only from this map/seed. Distances alone do not establish success.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--maps',default='C')
    p.add_argument('--seeds',default='0')
    p.add_argument('--modes',default='baseline,residual')
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--speed-limit-mps',type=float,default=3.0)
    p.add_argument('--steer-gain',type=float,default=1.0)
    p.add_argument('--kp-lateral',type=float,default=0.035)
    p.add_argument('--kp-heading',type=float,default=0.30)
    p.add_argument('--residual-limit',type=float,default=0.08)
    p.add_argument('--steer-limit',type=float,default=0.35)
    p.add_argument('--max-steer-delta',type=float,default=0.12)
    p.add_argument('--guard-lane-offset-m',type=float,default=1.5)
    p.add_argument('--guard-heading-rad',type=float,default=0.45)
    p.add_argument('--guard-road-margin-m',type=float,default=0.0,
                   help='0 disables approximate preemptive guard (default, closer to V2.9)')
    p.add_argument('--warn-road-margin-m',type=float,default=0.35)
    p.add_argument('--max-gap',type=float,default=2.0)
    p.add_argument('--trace-stride',type=int,default=5)
    p.add_argument('--tail-steps',type=int,default=25)
    p.add_argument('--out',type=Path,default=Path('agent4_v30_AB_C.json'))
    a=p.parse_args()
    names=[v.strip() for v in a.maps.split(',') if v.strip()]
    seeds=[int(v.strip()) for v in a.seeds.split(',') if v.strip()]
    modes=[v.strip() for v in a.modes.split(',') if v.strip()]
    if not names or not seeds or any(x<0 for x in seeds): p.error('Invalid maps/seeds')
    if not modes or len(set(modes))!=len(modes) or any(x not in ('baseline','residual') for x in modes):
        p.error('Modes must be baseline,residual or a non-repeated subset')
    if not (0<a.speed_limit_mps<=6 and 0<a.steer_gain<=3 and 0<a.steer_limit<=0.7 and
            0<a.max_steer_delta<=0.5 and 0<a.guard_lane_offset_m<=2 and
            0.1<a.guard_heading_rad<=math.pi/2 and 0<=a.guard_road_margin_m<=1 and
            0<=a.warn_road_margin_m<=1 and 0<a.max_gap<=5 and
            0<=a.kp_lateral<=0.5 and 0<=a.kp_heading<=2 and
            0<a.residual_limit<=0.35 and a.steps>=5 and
            1<=a.trace_stride<=500 and 1<=a.tail_steps<=200):
        p.error('Invalid speed/steer/feedback/guard/trace settings')
    model=joblib.load(a.model)
    if list(model.get('features',[]))!=FEATURES or model.get('label_mode')!='at_t':
        raise ValueError('Model schema mismatch: expected PDD V2 at_t features')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report={
        'status':'V30_AB_LEARNED_STEERING_PLUS_BOUNDED_FEEDBACK',
        'maps':names,'seeds':seeds,'modes':modes,
        'settings':{k:v for k,v in vars(a).items() if k not in ('model','out','maps','seeds','modes')},
        'model_path':str(a.model),'learned_steering_applied':True,
        'learned_longitudinal_applied':False,'idm_used':False,
        'residual_is_learned':False,'traffic_density':0.0,
        'episodes_by_map':[],
        'warnings':[
            'Only traffic-free MetaDrive simulation. Do NOT use on a physical vehicle.',
            'Residual gains are heuristic and NOT validated by the CARLA PDD model training.',
            'Comparing two closed-loop runs from one seed does not establish generalization.',
            'Road margin is an approximate geometric heuristic; MetaDrive flags are definitive.',
            'Both modes use model for steering, independent speed governor for longitudinal.',
        ],
    }
    for map_name in names:
        env=None; mode_results=[]; comparisons=[]
        try:
            env=MetaDriveEnv(dict(use_render=False,map=map_name,
                                  num_scenarios=max(seeds)+1,
                                  traffic_density=0.0,
                                  horizon=max(a.steps+10,1000),log_level=40))
            for seed in seeds:
                runs={}
                for mode in modes:
                    result=pilot_episode(env,seed,model,mode,a)
                    mode_results.append(result)
                    runs[mode]=result
                    print(f'map={map_name} seed={seed} mode={mode} status={result["status"]} '
                          f'steps={result["n_applied_steps"]} '
                          f'distance~{result["travelled_approx_m"]:.1f}m '
                          f'max_heading={math.degrees(result["max_abs_heading_error_rad"]):.1f}deg')
                compare=summarize_comparison(runs)
                if compare is not None:
                    comparisons.append({'seed':seed,**compare})
        finally:
            if env is not None: env.close()
        report['episodes_by_map'].append({'map':map_name,'episodes':mode_results,
                                          'comparisons':comparisons})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Saved:',a.out.resolve())
    print('Model steering applied: YES; bounded rule-based residual: only residual mode; '
          'learned longitudinal: NO; IDM: NO.')


if __name__=='__main__':
    main()
