"""Agent4 V2.7: *learned steering* closed-loop simulation pilot with independent speed governor.

Not an autonomous Agent 4 end-to-end evaluation: model supplies steering ONLY;
longitudinal input comes from a speed governor. No IDMPolicy, no teacher fallback.
Only for traffic-free MetaDrive simulation. Never use on a physical vehicle.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder, filename in [
    ('agent4_pdd_v2', 'shared_adapter_v2.py'),
    ('agent4_calibration_v22', 'coordinate_patch_v22.py'),
    ('agent4_calibration_v23', 'route_lookahead_v23.py'),
    ('agent4_calibration_v24', 'lane_route_v24.py'),
]:
    if not (ROOT / folder / filename).is_file():
        raise RuntimeError(f'Missing dependency: {ROOT / folder / filename}')
    sys.path.insert(0, str(ROOT / folder))

from shared_adapter_v2 import FEATURES, metadrive_features  # noqa: E402
from coordinate_patch_v22 import determine_local_y_to_right_multiplier, correct_features_to_pdd_right  # noqa: E402
from route_lookahead_v23 import collect_forward_segments  # noqa: E402
from lane_route_v24 import choose_lane_chain, sample_lane_chain  # noqa: E402


def clamp(x, a, b):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f'Nonfinite action: {x}')
    return max(float(a), min(float(b), x))


def adaptive_horizons(remaining_m, far=(25.0,45.0)):
    """Keep training lookahead whenever possible; shrink strictly near destination.

    Returns (horizons, is_shortened). Do NOT silently pretend adaptive targets
    are equivalent to PDD training target-point distribution.
    """
    remaining = float(remaining_m)
    if not math.isfinite(remaining) or remaining <= 0.5:
        raise ValueError('Remaining route too short for reliable model input')
    near, distant = [float(v) for v in far]
    if not 0 < near < distant:
        raise ValueError('Invalid training horizons')
    if remaining >= distant + 1e-5:
        return (near,distant),False
    far_eff = 0.90 * remaining
    near_eff = min(near, 0.55 * remaining, 0.65 * far_eff)
    if not 0 < near_eff < far_eff < remaining:
        raise ValueError('Invalid shortened waypoints')
    return (near_eff,far_eff),True


def remaining_route_m(chain):
    return sum(max(0.0, float(item['lane'].length)-float(item['start_longitude'])) for item in chain)


def governed_speed_action(speed_mps, speed_limit_mps, remaining_m,
                          kp=0.23, accel_limit=0.4, brake_limit=0.40):
    """Simple independent speed controller; not a learned longitudinal output."""
    speed=float(speed_mps)
    if not (math.isfinite(speed) and speed >= 0):
        raise ValueError('Invalid speed')
    remaining=float(remaining_m)
    if remaining<45:
        target=min(float(speed_limit_mps),max(0.65,remaining*0.16))
    else:
        target=float(speed_limit_mps)
    raw=float(kp)*(target-speed)
    return clamp(raw,-brake_limit,accel_limit),target


def bounded_steering(carla_steer, gain=1.0, steer_limit=0.35,
                     previous=None, max_delta=0.12):
    raw=-float(carla_steer)*float(gain)  # confirmed CARLA(+right)->MetaDrive(+left)
    steer=clamp(raw,-steer_limit,steer_limit)
    if previous is not None:
        steer=clamp(steer,previous-max_delta,previous+max_delta)
    return steer


def feature_for_model(env, min_gap=2.0):
    agent=env.agent
    sign,_=determine_local_y_to_right_multiplier(agent)
    f0,_=metadrive_features(env)
    f=correct_features_to_pdd_right(f0,sign)
    segments=collect_forward_segments(agent.navigation,agent)
    chain,gaps=choose_lane_chain(segments,agent.position,max_gap_m=min_gap)
    if gaps:
        raise ValueError(f'Lane-chain gap in planned route: {gaps[:2]}')
    remaining=remaining_route_m(chain)
    horizons,shortened=adaptive_horizons(remaining)
    pxy=[float(v) for v in agent.position[:2]]
    coords=[]
    for lookahead in horizons:
        point,diag=sample_lane_chain(chain,lookahead)
        if not diag['complete']:
            raise ValueError(f'Unexpected horizon incomplete: {diag}')
        x,y=agent.convert_to_local_coordinates((point[0]-pxy[0],point[1]-pxy[1]),0.0)
        x=float(x); y=float(y)*sign
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('Invalid local waypoint')
        coords.append([x,y])
    f.update(dict(wp1_forward_m=coords[0][0],wp1_right_m=coords[0][1],
                  wp2_forward_m=coords[1][0],wp2_right_m=coords[1][1]))
    if not all(math.isfinite(float(f[k])) for k in FEATURES):
        raise ValueError('Nonfinite model feature')
    return f, {'remaining_m':remaining, 'effective_lookahead_m':list(horizons),
               'adapted':shortened, 'lane_indices':[v['lane_index'] for v in chain],
               'lane_chain_continuous':True}


def predict_steering(model,features):
    x=np.asarray([[features[k] for k in FEATURES]],dtype=float)
    steer=float(model['models']['steering'].predict(x)[0])
    if not math.isfinite(steer):
        raise ValueError('Nonfinite steering prediction')
    return steer


def lane_offset(agent):
    lane=getattr(agent,'lane',None)
    if lane is None:
        return None,None
    try:
        _,lateral=lane.local_coordinates(agent.position)
        return float(lateral),float(lane.width)
    except (AttributeError,TypeError,ValueError):
        return None,None


def episode(env, seed, steps, model, speed_limit, gain, steer_limit,
            max_delta, guard_offset, max_gap, trace_stride):
    observations=[]
    counters={'model_steering_steps':0,'adaptive_lookahead_steps':0,
              'steering_limit_steps':0,'speed_governor_steps':0}
    status='step_limit'; end_info=None; errors=[]; travel=0.0; previous=0.0
    try:
        env.reset(seed=seed)
        agent=env.agent
        for t in range(steps):
            offset,width=lane_offset(agent)
            if offset is not None and abs(offset)>guard_offset:
                status='lane_departure_guard';break
            try:
                f,diag=feature_for_model(env,max_gap)
                steer_carla=predict_steering(model,f)
                steer=bounded_steering(steer_carla,gain,steer_limit,previous,max_delta)
                longitudinal,target=governed_speed_action(f['speed_mps'],speed_limit,diag['remaining_m'])
            except Exception as exc:
                status='observation_or_model_guard'
                errors.append(f'{type(exc).__name__}: {exc}')
                break
            action=[float(steer),float(longitudinal)]
            # This is the FIRST and ONLY learned-action actuation (steering only).
            # MetaDrive must be using the default external-action policy, not IDMPolicy.
            before_xy=np.asarray(agent.position[:2],dtype=float)
            try:
                _,reward,terminated,truncated,info=env.step(action)
            except Exception as exc:
                status='env_step_error'
                errors.append(f'{type(exc).__name__}: {exc}')
                break
            after_xy=np.asarray(env.agent.position[:2],dtype=float)
            travel+=float(np.linalg.norm(after_xy-before_xy))
            counters['model_steering_steps']+=1
            counters['adaptive_lookahead_steps']+=int(diag['adapted'])
            counters['steering_limit_steps']+=int(abs(steer_carla*gain)>steer_limit)
            counters['speed_governor_steps']+=1
            if t%trace_stride==0 or terminated or truncated or diag['adapted'] and t%5==0:
                observations.append({'step':t,'speed_mps_before':f['speed_mps'],
                                     'target_speed_mps':target,
                                     'steer_carla_pred':steer_carla,
                                     'action_applied':action,
                                     'lane_offset_m_before':offset,
                                     'lane_width_m':width,
                                     'remaining_route_m':diag['remaining_m'],
                                     'lookahead_m':diag['effective_lookahead_m'],
                                     'adapted':diag['adapted'],
                                     'reward':float(reward),
                                     'xy_after':after_xy.tolist()})
            previous=steer
            agent=env.agent
            if terminated or truncated:
                end_info={'terminated':bool(terminated),'truncated':bool(truncated),
                          'arrive_dest':bool(info.get('arrive_dest',False)),
                          'out_of_road':bool(info.get('out_of_road',False)),
                          'crash':bool(info.get('crash',False))}
                if end_info['arrive_dest']:
                    status='arrive_dest'
                elif end_info['out_of_road']:
                    status='out_of_road'
                elif end_info['crash']:
                    status='crash'
                else:
                    status='terminated_or_truncated'
                break
    except Exception as exc:
        status='episode_error';errors.append(f'{type(exc).__name__}: {exc}')
    return {'seed':seed,'status':status,'end_info':end_info,'travelled_approx_m':travel,
            'counts':counters,'errors':errors,'trace':observations}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--maps',default='S')
    ap.add_argument('--seeds',default='0')
    ap.add_argument('--steps',type=int,default=650)
    ap.add_argument('--speed-limit-mps',type=float,default=3.0)
    ap.add_argument('--steer-gain',type=float,default=1.0)
    ap.add_argument('--steer-limit',type=float,default=0.35)
    ap.add_argument('--max-steer-delta',type=float,default=0.12)
    ap.add_argument('--guard-lane-offset-m',type=float,default=1.5)
    ap.add_argument('--max-gap',type=float,default=2.0)
    ap.add_argument('--trace-stride',type=int,default=5)
    ap.add_argument('--out',type=Path,default=Path('agent4_v27_guarded_pilot.json'))
    a=ap.parse_args()
    maps=[s.strip() for s in a.maps.split(',') if s.strip()]
    seeds=[int(s.strip()) for s in a.seeds.split(',') if s.strip()]
    if not maps or not seeds or min(seeds)<0 or a.steps<5 or a.trace_stride<1:
        ap.error('Invalid maps/seeds/steps/trace-stride')
    if not (0<a.speed_limit_mps<=6 and 0<a.steer_gain<=3 and
            0<a.steer_limit<=0.7 and 0<a.max_steer_delta<=0.5 and
            0<a.guard_lane_offset_m<=2 and 0<a.max_gap<=5):
        ap.error('Invalid bounded pilot settings')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    model=joblib.load(a.model)
    if list(model.get('features',[]))!=FEATURES or model.get('label_mode')!='at_t':
        raise ValueError('Model schema mismatch with PDD V2 at_t')
    result={'status':'V27_GUARDED_LEARNED_STEERING_WITH_INDEPENDENT_SPEED_CONTROL',
            'learned_steering_applied':True,'learned_longitudinal_applied':False,
            'teacher_idm_used_for_control':False,
            'this_is_full_agent4_autonomy':False,
            'traffic_density':0.0,
            'maps':maps,'seeds':seeds,
            'parameters':{'steps':a.steps,'speed_limit_mps':a.speed_limit_mps,
                          'steer_gain':a.steer_gain,'steer_limit':a.steer_limit,
                          'max_steer_delta':a.max_steer_delta,
                          'guard_lane_offset_m':a.guard_lane_offset_m,
                          'max_gap':a.max_gap},
            'results':[], 'warnings':[
              'Hybrid pilot: model controls steering; speed comes from independent governor.',
              'Steering gain 1 is experimental, not physically calibrated.',
              'Adaptive near-destination lookahead differs from PDD training data.',
              'Only traffic-free MetaDrive simulation; NOT real-road safe.',
              'Travel, success and failures are not comparable to IDM under same speed policy.',
            ]}
    for map_name in maps:
        env=None; episodes=[];map_errors=[]
        try:
            env=MetaDriveEnv(dict(use_render=False, map=map_name,
                                  num_scenarios=max(seeds)+1, traffic_density=0.0,
                                  horizon=max(a.steps+10,1000), log_level=40))
            for seed in seeds:
                outcome=episode(env,seed,a.steps,model,a.speed_limit_mps,
                                a.steer_gain,a.steer_limit,a.max_steer_delta,
                                a.guard_lane_offset_m,a.max_gap,a.trace_stride)
                episodes.append(outcome)
                print(f"  map={map_name} seed={seed} status={outcome['status']} "
                      f"steps={outcome['counts']['model_steering_steps']} "
                      f"distance~{outcome['travelled_approx_m']:.1f}m")
        except Exception as exc:
            map_errors.append(f'{type(exc).__name__}: {exc}')
        finally:
            if env is not None:
                try:env.close()
                except Exception as exc:map_errors.append(f'close {type(exc).__name__}: {exc}')
        result['results'].append({'map':map_name,'episodes':episodes,'errors':map_errors,
                                  'n_arrived':sum(e['status']=='arrive_dest' for e in episodes),
                                  'n_guarded':sum(e['status'].endswith('_guard') for e in episodes)})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('Saved:',a.out.resolve())
    print('Model steering applied: YES. Learned longitudinal applied: NO. No IDM fallback.')


if __name__=='__main__':
    main()
