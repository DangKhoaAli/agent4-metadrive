"""Controlled moving lead in MetaDrive, observability checks and traffic action hooks.

For first integration use S maps (long initial straight). All lead scenarios are scripted;
this is NOT random public-road traffic. If MetaDrive 0.4.3 API differs, fail loudly.
"""
from __future__ import annotations
import math,sys
from pathlib import Path
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_longitudinal_v36'))
from common_v36 import make_row
sys.path.insert(0,str(ROOT/'agent4_robustness_v34'))
from benchmark_v34 import v30,eval_one
from common_v38 import SCENARIOS,front_features,feature_names,expert_follow,predict


def phase_speed(scenario,step):
    if scenario=='slow_release': return 1.0 if step<170 else 5.0
    if scenario=='stop_release': return 0.0 if step<110 else 5.0
    if scenario=='brake_release': return 2.0 if step<45 else (0.0 if step<140 else 5.0)
    raise ValueError(f'Unsupported scenario {scenario}')


def scheduled_lead_longitudinal(speed_mps, target_mps):
    """Signed MetaDrive action for a scripted lead speed (metres/second).

    IDMPolicy may reset target_speed while selecting its lane; its action[1]
    therefore cannot reliably enforce a low scripted speed. Use IDM steering
    but independently control the lead's longitudinal speed.
    """
    speed=max(0.,float(speed_mps))
    target=max(0.,float(target_mps))
    if not (math.isfinite(speed) and math.isfinite(target)):
        raise ValueError('Invalid lead speed')
    if target<=0.0:
        return -0.6 if speed>0.10 else -0.15
    return max(-0.6,min(0.4,0.30*(target-speed)))


def controlled_lead_class():
    from metadrive.policy.idm_policy import IDMPolicy
    class ControlledLeadPolicy(IDMPolicy):
        def __init__(self,control_object,random_seed,scenario):
            super().__init__(control_object,random_seed)
            self.scenario=scenario
            self.tick=0
            self.enable_lane_change=False
        def act(self,*args,**kwargs):
            target=phase_speed(self.scenario,self.tick)
            self.tick+=1
            # IDM for lateral lane following only; separate deterministic speed schedule.
            action=super().act(*args,**kwargs)
            action=[float(action[0]),scheduled_lead_longitudinal(self.control_object.speed,target)]
            self.action_info['action']=action
            return action
    return ControlledLeadPolicy


def resolve_lead_spawn_slot(ego, road_network, gap_m, end_buffer_m=2.0):
    """Walk the ego navigation route, not just its 10 m initial road segment.

    Returns (lane_index, spawn_longitude). Each road's longitude restarts at 0;
    gap_m measures centre-to-centre forward distance along connected lanes.
    Fails closed when a road does not provide the same lane number.
    """
    lane=ego.lane
    if lane is None:
        raise RuntimeError('Ego lane unavailable at reset')
    lane_index=tuple(ego.lane_index)
    if len(lane_index)!=3:
        raise RuntimeError(f'Unexpected MetaDrive lane index: {lane_index!r}')
    navigation=getattr(ego,'navigation',None)
    checkpoints=list(getattr(navigation,'checkpoints',[]) or [])
    if len(checkpoints)<2:
        raise RuntimeError('Ego navigation checkpoints unavailable; cannot safely place lead ahead')
    try:
        start=next(i for i in range(len(checkpoints)-1)
                   if (checkpoints[i],checkpoints[i+1])==lane_index[:2])
    except StopIteration as exc:
        raise RuntimeError(f'Ego lane {lane_index!r} missing from navigation route {checkpoints!r}') from exc
    graph=getattr(road_network,'graph',None)
    if graph is None:
        raise RuntimeError('MetaDrive road_network.graph unavailable')
    remaining=float(gap_m)
    if not math.isfinite(remaining) or remaining<=0:
        raise ValueError('gap_m must be finite and positive')
    lane_no=int(lane_index[2])
    ego_s=float(lane.local_coordinates(ego.position)[0])
    if not math.isfinite(ego_s):
        raise RuntimeError('Invalid longitudinal coordinate of ego')
    for i in range(start,len(checkpoints)-1):
        a,b=checkpoints[i:i+2]
        try:
            lanes=graph[a][b]
        except (KeyError,TypeError) as exc:
            raise RuntimeError(f'Missing road along navigation route: {a!r}->{b!r}') from exc
        if lane_no >= len(lanes):
            raise RuntimeError(f'Route road {a!r}->{b!r} has only {len(lanes)} lanes; '
                               f'ego uses lane {lane_no}. Refusing an unrequested lane change.')
        target_lane=lanes[lane_no]
        length=float(target_lane.length)
        current_s=ego_s if i==start else 0.0
        available=length-current_s
        if available<0:
            raise RuntimeError(f'Invalid route geometry at {a!r}->{b!r}: '
                               f's={current_s:.2f} > length={length:.2f}')
        # Do not put a stationary/slow car exactly at the end of a road segment.
        if remaining<=available-end_buffer_m:
            longitudinal=current_s+remaining
            chosen=(a,b,lane_no)
            # Verify the same lane is used by MetaDrive's spawn config.
            if not math.isfinite(longitudinal):
                raise RuntimeError('Invalid lead spawn longitude')
            return chosen, longitudinal
        remaining-=available
        if remaining<=0:
            # Target falls in the last 2 m of this road. Place on the next
            # road just after its start (same centre-to-centre gap).
            remaining=max(0.,remaining)
    raise RuntimeError('Not enough FORWARD route length to spawn lead '
                       f'{gap_m:.1f} m ahead (ego_s={ego_s:.1f}, '
                       f'current_lane_length={lane.length:.1f}, '
                       f'checkpoints={len(checkpoints)}); choose a longer map')


def spawn_lead(env,scenario,gap_m=24.,seed=0):
    if scenario not in SCENARIOS:raise ValueError('Invalid scenario')
    if not 12<=gap_m<=35:raise ValueError('Gap must be 12..35 m')
    ego=env.agent
    road_network=env.current_map.road_network
    spawn_lane_index,spawn_longitude=resolve_lead_spawn_slot(ego,road_network,gap_m)
    from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle
    manager=getattr(env.engine,'traffic_manager',None)
    if manager is None or not hasattr(manager,'_traffic_vehicles'):
        raise RuntimeError('Required MetaDrive TrafficManager API unavailable')
    lead=manager.spawn_object(TrafficDefaultVehicle,
            vehicle_config={'spawn_lane_index':spawn_lane_index,
                            'spawn_longitude':spawn_longitude,'enable_reverse':False})
    manager.add_policy(lead.id,controlled_lead_class(),lead,int(seed)+50101,scenario)
    manager._traffic_vehicles.append(lead)
    if not any(v is lead for v in manager.traffic_vehicles):
        raise RuntimeError('Lead was not registered with TrafficManager')
    # At the requested spawn, make sure its physical placement is downstream
    # on the same trajectory, not on another road/lane with the same node name.
    reported_index=getattr(lead,'lane_index',None)
    lead_lane=tuple(reported_index) if reported_index is not None else None
    if lead_lane is not None and lead_lane!=tuple(spawn_lane_index):
        raise RuntimeError(f'Lead spawned on {lead_lane!r}, expected {spawn_lane_index!r}')
    return lead


def observe_front(env,radius=80.):
    """Observe by scanning active traffic, never using the scenario ID or lead policy phase."""
    ego=env.agent
    tm=getattr(env.engine,'traffic_manager',None)
    if tm is None:raise RuntimeError('Traffic manager missing')
    best=None
    for vehicle in tm.traffic_vehicles:
        if vehicle is ego:continue
        try:
            forward,lateral=ego.convert_to_local_coordinates(vehicle.position,ego.position)
            fwd=float(forward);side=float(lateral)
            if not (math.isfinite(fwd) and math.isfinite(side)):continue
            if fwd<=0 or fwd>radius+6 or abs(side)>min(2.0,float(ego.lane.width)*.49):continue
            # Require roughly aligned heading; reject crossing/opposing traffic.
            if abs(math.atan2(math.sin(float(vehicle.heading_theta)-float(ego.heading_theta)),
                              math.cos(float(vehicle.heading_theta)-float(ego.heading_theta))))>.35:
                continue
            gap=max(0.,fwd-.5*(float(ego.LENGTH)+float(vehicle.LENGTH)))
            if gap>radius:continue
            heading=ego.heading
            closing=max(0.,float((ego.velocity[0]-vehicle.velocity[0])*heading[0]
                                  +(ego.velocity[1]-vehicle.velocity[1])*heading[1]))
            candidate={'present':True,'gap_m':gap,'speed_mps':max(0.,float(vehicle.speed)),
                       'closing_mps':closing}
            if best is None or candidate['gap_m']<best['gap_m']:best=candidate
        except (ValueError,TypeError,AttributeError):
            continue
    return best if best is not None else {'present':False,'gap_m':radius,
            'speed_mps':0.,'closing_mps':0.}


@contextmanager
def reset_with_lead(env,scenario,gap,seed):
    """Intercept one call to env.reset inside V3.4/V3.5's proven episode loop."""
    real_reset=env.reset
    def reset(*a,**k):
        ret=real_reset(*a,**k)
        # Ensure same seed is used for map reset and lead policy randomness.
        kseed=k.get('seed',seed)
        spawn_lead(env,scenario,gap_m=gap,seed=kseed)
        initial=observe_front(env)
        if not initial['present'] or abs(initial['gap_m']-gap)>6:
            raise RuntimeError(f'Controlled lead not observed correctly at reset: {initial}')
        return ret
    env.reset=reset
    try:yield
    finally:env.reset=real_reset


@contextmanager
def captured_longitudinal(env,pdd_features,mode,old_bundle,new_bundle,records):
    """Identical steering and traffic; teacher and students differ only in longitudinal action."""
    if mode not in ('teacher_v38','learned_v37','learned_v38'):
        raise ValueError(mode)
    real_f=v30.feature_for_model;real_g=v30.planned_geometry;real_long=v30.governed_speed_action
    cache={}
    def observed_features(sim,max_gap):
        feat,route=real_f(sim,max_gap)
        cache.update(features=feat,route=route)
        return feat,route
    def observed_geom(car,max_gap):
        geom=real_g(car,max_gap)
        cache['geom']=geom
        return geom
    def longitudinal(speed,base_speed,remaining):
        governor,route_target=real_long(speed,base_speed,remaining)
        if not all(k in cache for k in ('features','route','geom')):
            raise RuntimeError('Pre-step state missing')
        front=observe_front(env)
        base=make_row(cache['features'],pdd_features,cache['geom'],
            cache['route']['adapted'],route_target,cache['route']['remaining_m'])
        row={**base,**front_features(front)}
        teacher,teacher_target=expert_follow(speed,route_target,front)
        if mode=='teacher_v38':action=teacher
        elif mode=='learned_v37':action=predict(old_bundle,base)
        else:action=predict(new_bundle,row)
        records.append(dict(step=len(records),front=front,
            route_target_mps=float(route_target),teacher_target_mps=float(teacher_target),
            teacher_action=float(teacher),applied_longitudinal=float(action),
            baseline_governor_action=float(governor),row=row))
        return float(action),float(teacher_target if mode=='teacher_v38' else route_target)
    v30.feature_for_model=observed_features
    v30.planned_geometry=observed_geom
    v30.governed_speed_action=longitudinal
    try:yield
    finally:
        v30.feature_for_model=real_f
        v30.planned_geometry=real_g
        v30.governed_speed_action=real_long


def run_one(env,map_name,seed,speed,scenario,gap,mode,pdd,steering,old,new,steps,stride=30):
    actions=[]
    with reset_with_lead(env,scenario,gap,seed),captured_longitudinal(
            env,pdd['features'],mode,old,new,actions):
        result=eval_one(env,map_name,seed,speed,'student_v33',pdd,steering,steps,stride)
    n=result['summary']['n_applied_steps']
    if len(actions)!=n:
        raise RuntimeError(f'Action alignment mismatch: {len(actions)} vs {n}, '
                           f'episode status {result["summary"]["status"]}')
    for item in result['trace']:
        a=actions[item['step']]
        if abs(item['action_applied'][1]-a['applied_longitudinal'])>1e-8:
            raise RuntimeError('Applied action != recorded traffic action')
        item['traffic_v38']={k:a[k] for k in ('front','teacher_action','applied_longitudinal',
                      'route_target_mps','teacher_target_mps')}
    active=[a for a in actions if a['front']['present']]
    summary=result['summary']
    summary.update(profile=mode,scenario=scenario,gap_spawn_m=gap,
                   learned_longitudinal_applied=mode!='teacher_v38',
                   longitudinal_prediction_calls=(n if mode!='teacher_v38' else 0),
                   longitudinal_actions_aligned=True,
                   front_seen_steps=len(active),
                   front_close_steps=sum(a['front']['gap_m']<18 for a in active),
                   teacher_intervention_steps=sum(a['teacher_target_mps']<a['route_target_mps']-.1 for a in actions),
                   braking_steps=sum(a['applied_longitudinal']<-.02 for a in actions),
                   min_front_gap_m=min((a['front']['gap_m'] for a in active),default=None),
                   min_ttc_s=min((front_features(a['front'])['front_ttc_s'] for a in active),default=None),
                   following_error_mae=sum(abs(a['teacher_action']-a['applied_longitudinal']) for a in actions)/n if n else None)
    result.update(profile=mode,scenario=scenario,gap_spawn_m=gap)
    return result,actions
