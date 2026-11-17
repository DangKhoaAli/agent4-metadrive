"""Agent 4 V3.12B: verified full-step IDM demonstrations for MetaDrive 0.4.3.

Collects pre-action state and the *actual vehicle current_action* AFTER a policy
step. Does not train a model. Must run inside the user's installed MetaDrive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

from geometry_v312 import compare_vehicle_route, approximate_vehicle_clearance, finite, lane_heading, wrap_angle

VERSION = 'AGENT4_V312B_FULL_STEP_IDM_DATASET'
MAPS = ('S', 'SSSSS', 'C', 'X')
STEP_S = 0.1
TOL = 5e-4

FIELDS = [
    'version','split','map','seed','target_speed_mps','step','time_s',
    'pos_x_m','pos_y_m','heading_rad','speed_mps',
    'vehicle_lane_id','vehicle_lane_class','planned_lane_id','planned_lane_class',
    'next_lane_id','next_lane_class','selected_next_lane_id','selected_next_lane_seam_m','planned_lateral_right_m','heading_error_rad',
    'lane_s_m','distance_to_lane_end_m','curvature_proxy_rad_per_m',
    'heading_change_ahead_rad','route_heading_jump_at_transition_rad',
    'lookahead_heading_5m_rad','lookahead_heading_10m_rad','lookahead_heading_20m_rad',
    'lookahead_curvature_5m','lookahead_curvature_10m','lookahead_curvature_20m',
    'lookahead_5m_available','lookahead_10m_available','lookahead_20m_available',
    'edge_clearance_proxy_m','left_side_m','right_side_m',
    'physical_lane_transition','planned_lane_transition','is_circular_lane',
    'raw_steering','raw_throttle_brake','applied_steering','applied_throttle_brake',
    'expected_clipped_steering','expected_clipped_throttle_brake',
    'raw_out_of_range','raw_action_clipped','current_action_verified',
    'label_valid','trainable_row','excluded_reason',
    'post_speed_mps','post_pos_x_m','post_pos_y_m',
    'arrive_dest','out_of_road','crash','on_yellow_continuous_line','terminated','truncated',
]


def parse_seed_spec(spec):
    values=[]
    if not spec.strip(): return []
    for term in spec.split(','):
        token=term.strip()
        if '-' in token:
            parts=token.split('-')
            if len(parts)!=2 or not all(x.isdigit() for x in parts):
                raise ValueError(f'Invalid seed range {token!r}')
            a,b=map(int,parts)
            if a>b or b-a>10000: raise ValueError('Seed range reversed or too large')
            values.extend(range(a,b+1))
        elif token.isdigit(): values.append(int(token))
        else: raise ValueError(f'Invalid seed {token!r}')
    if len(values)!=len(set(values)): raise ValueError('Duplicate seed')
    return values


def parse_maps(spec):
    values=[v.strip() for v in spec.split(',') if v.strip()]
    if not values or len(values)!=len(set(values)) or any(v not in MAPS for v in values):
        raise ValueError(f'Maps must be unique and among {MAPS}')
    return values


def parse_speeds(spec):
    values=[]
    for token in spec.split(','):
        try: v=float(token.strip())
        except ValueError: raise ValueError('Speeds must be comma separated numbers') from None
        if not math.isfinite(v) or not 1.0<=v<=12.0: raise ValueError('Speed out of [1,12] m/s')
        values.append(v)
    if not values or len(set(values))!=len(values): raise ValueError('Duplicate or missing speeds')
    return values


def split_seed_lists(args):
    splits={'train':parse_seed_spec(args.train_seeds),
            'val':parse_seed_spec(args.val_seeds),
            'test':parse_seed_spec(args.test_seeds)}
    all_items=[s for values in splits.values() for s in values]
    if len(all_items)!=len(set(all_items)) or not all_items:
        raise ValueError('Train/val/test seeds must be disjoint, with at least one seed')
    return splits


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def atom_json(path,doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(doc,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def get_policy(env):
    for key in (getattr(env.agent,'id',None),getattr(env.agent,'name',None)):
        if key is None:continue
        try:
            result=env.engine.get_policy(key)
            if result is not None:return result
        except (KeyError,TypeError,AttributeError):pass
    raise RuntimeError('No ego policy found; refusing to use placeholder env.step action')


def make_policy(normal_speed_kmh):
    from metadrive.policy.idm_policy import IDMPolicy
    class RecordedIDMPolicy(IDMPolicy):
        NORMAL_SPEED=float(normal_speed_kmh)
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.record_count=0
            self.last_recorded_action=None
        def act(self,*args,**kwargs):
            raw=super().act(*args,**kwargs)
            a=checked_pair(raw,'IDM raw policy action')
            self.record_count+=1
            self.last_recorded_action=a
            return raw
    return RecordedIDMPolicy


def checked_pair(obj,description):
    try: values=[float(x) for x in obj]
    except (ValueError,TypeError,OverflowError) as e:raise ValueError(f'{description} unreadable') from e
    if len(values)!=2 or not all(math.isfinite(v) for v in values):
        raise ValueError(f'{description} invalid: {values!r}')
    return values


def clip(v):
    return max(-1.0,min(1.0,float(v)))


def read_applied_action(agent,raw):
    """Fail closed unless vehicle state confirms processed commands.

    MetaDrive's BaseVehicle.before_step calls _preprocess_action and stores the
    result in last_current_action/current_action, then _set_action sets steering
    and throttle_brake. The assert against clip(raw) guards version changes.
    """
    a=checked_pair(getattr(agent,'current_action',None),'vehicle.current_action')
    steering=finite(getattr(agent,'steering',None))
    throttle=finite(getattr(agent,'throttle_brake',None))
    if steering is None or throttle is None:
        raise RuntimeError('Cannot inspect vehicle applied steering/throttle')
    if max(abs(a[0]-steering),abs(a[1]-throttle))>TOL:
        raise RuntimeError(f'current_action != vehicle controls: {a}, {steering,throttle}')
    expected=[clip(x) for x in raw]
    if max(abs(a[0]-expected[0]),abs(a[1]-expected[1]))>TOL:
        raise RuntimeError(f'current_action does not match clipped IDM raw: raw={raw}, action={a}')
    if any(abs(x)>1+TOL for x in a):
        raise RuntimeError('Vehicle applied action not normalized [-1,1]')
    return a


def choose_connected_next_lane(lane, candidates, maximum_gap_m=2.0):
    """Pick the geometrically connected continuation, not next_ref_lanes[0].

    Parallel lanes can have distinct curvature/radius. This prevents lane 2's
    lookahead from accidentally following lane 0 in the next road block.
    """
    if lane is None or not candidates:return None,None
    try:
        p=lane.position(float(lane.length),0.0)
        end=(float(p[0]),float(p[1]))
    except (TypeError,ValueError,AttributeError,IndexError):return None,None
    ranked=[]
    for n in candidates:
        try:
            q=n.position(0.0,0.0)
            gap=math.hypot(float(q[0])-end[0],float(q[1])-end[1])
            if math.isfinite(gap):
                same_index=int(getattr(lane,'index',None) is not None and
                               getattr(n,'index',None) is not None and
                               lane.index[-1]==n.index[-1])
                ranked.append((gap,-same_index,n))
        except (TypeError,ValueError,AttributeError,IndexError):continue
    if not ranked:return None,None
    gap,_,winner=min(ranked,key=lambda t:(round(t[0],6),t[1]))
    return (winner,gap) if gap<=maximum_gap_m else (None,gap)


def lookahead_features(agent,distances=(5.,10.,20.)):
    """Measure heading across a *connected* route segment when available.

    Never extrapolates beyond two road segments. Missing horizons are explicit.
    """
    fields={}
    nav=getattr(agent,'navigation',None)
    current=[l for l in (getattr(nav,'current_ref_lanes',None) or []) if l is not None]
    next_lanes=[l for l in (getattr(nav,'next_ref_lanes',None) or []) if l is not None]
    lane=None
    if current:
        try:lane=min(current,key=lambda l:abs(float(l.local_coordinates(agent.position)[1])))
        except (TypeError,ValueError,AttributeError):lane=None
    connected,gap=choose_connected_next_lane(lane,next_lanes)
    fields['selected_next_lane_id']=repr(getattr(connected,'index',None)) if connected is not None else None
    fields['selected_next_lane_seam_m']=gap
    for d in distances:
        key=str(int(d))
        change=None;available=False
        try:
            if lane is None:raise ValueError('No planned lane')
            s=float(lane.local_coordinates(agent.position)[0]);length=float(lane.length)
            start_s=max(0.,min(length,s))
            current_heading=lane_heading(lane,start_s)
            horizon_s=start_s+d
            if horizon_s<=length:
                heading=lane_heading(lane,horizon_s);available=True
            elif connected is not None and horizon_s-length<=float(connected.length):
                heading=lane_heading(connected,horizon_s-length);available=True
            else:heading=None
            if available:change=wrap_angle(heading-current_heading)
        except (TypeError,ValueError,AttributeError,IndexError,KeyError):pass
        fields[f'lookahead_heading_{key}m_rad']=change
        fields[f'lookahead_curvature_{key}m']=change/d if change is not None else None
        fields[f'lookahead_{key}m_available']=int(available)
    return fields


def state_before_action(agent):
    g=compare_vehicle_route(agent)
    pose=g.get('planned') or {}
    xy=[float(x) for x in agent.position[:2]]
    speed=finite(getattr(agent,'speed_km_h',None))
    if speed is None or not g.get('route_geometry_available') or not pose:
        raise RuntimeError('Missing pre-step speed or planned lane geometry')
    lateral=finite(pose.get('lateral_right_m'))
    heading_err=finite(pose.get('heading_error_rad'))
    curvature=finite(pose.get('lane_curvature_proxy_rad_per_m'))
    if any(v is None for v in (lateral,heading_err,curvature)):
        raise RuntimeError('Nonfinite pre-step lane geometry')
    row={
        'pos_x_m':xy[0],'pos_y_m':xy[1],'heading_rad':finite(getattr(agent,'heading_theta',None)),
        'speed_mps':speed/3.6,'vehicle_lane_id':g.get('vehicle_lane_id'),
        'vehicle_lane_class':g.get('vehicle_lane_class'),
        'planned_lane_id':g.get('planned_lane_id'),
        'planned_lane_class':g.get('planned_lane_class'),
        'next_lane_id':g.get('planned_next_lane_id'),'next_lane_class':g.get('planned_next_lane_class'),
        'planned_lateral_right_m':lateral,'heading_error_rad':heading_err,
        'lane_s_m':finite(pose.get('lane_longitudinal_m')),
        'distance_to_lane_end_m':finite(pose.get('distance_to_lane_end_m')),
        'curvature_proxy_rad_per_m':curvature,
        'heading_change_ahead_rad':finite(pose.get('lane_heading_change_ahead_rad')),
        'route_heading_jump_at_transition_rad':finite(g.get('route_heading_jump_at_transition_rad')),
        'edge_clearance_proxy_m':approximate_vehicle_clearance(agent,heading_err),
        'left_side_m':finite(getattr(agent,'dist_to_left_side',None)),
        'right_side_m':finite(getattr(agent,'dist_to_right_side',None)),
        'is_circular_lane':int('CircularLane' in (g.get('planned_lane_class') or '')),
    }
    row.update(lookahead_features(agent))
    if row['heading_rad'] is None:raise RuntimeError('Missing heading_rad')
    return row


def post_state(agent):
    xy=[float(v) for v in agent.position[:2]]
    speed=finite(getattr(agent,'speed_km_h',None))
    return dict(post_speed_mps=speed/3.6 if speed is not None else None,
                post_pos_x_m=xy[0],post_pos_y_m=xy[1])


def make_step_row(pre,raw,applied,split,map_name,seed,target_speed,step,info,terminated,truncated,
                  physical_transition=False,planned_transition=False):
    bad=any(abs(x)>1+TOL for x in raw)
    reasons=[]
    if bad:reasons.append('raw_out_of_range')
    if not all(int(pre.get(f'lookahead_{d}m_available',0)) for d in (5,10)):
        reasons.append('lookahead_5_10_missing')
    if (pre.get('edge_clearance_proxy_m') is None):reasons.append('clearance_missing')
    row={x:None for x in FIELDS}
    row.update(pre)
    row.update(version=VERSION,split=split,map=map_name,seed=seed,
               target_speed_mps=target_speed,step=step,time_s=round(step*STEP_S,4),
               raw_steering=raw[0],raw_throttle_brake=raw[1],
               applied_steering=applied[0],applied_throttle_brake=applied[1],
               expected_clipped_steering=clip(raw[0]),expected_clipped_throttle_brake=clip(raw[1]),
               raw_out_of_range=int(bad),raw_action_clipped=int(any(abs(x-y)>TOL for x,y in zip(raw,applied))),
               current_action_verified=1,label_valid=1,trainable_row=int(not reasons),
               excluded_reason=';'.join(reasons),
               physical_lane_transition=int(physical_transition),
               planned_lane_transition=int(planned_transition),
               arrive_dest=int(bool(info.get('arrive_dest'))),out_of_road=int(bool(info.get('out_of_road'))),
               crash=int(any(bool(info.get(k)) for k in ('crash','crash_vehicle','crash_object','crash_building','crash_sidewalk'))),
               on_yellow_continuous_line=int(bool(info.get('on_yellow_continuous_line'))),
               terminated=int(bool(terminated)),truncated=int(bool(truncated)))
    return row


def episode_rel_path(split,m,seed,speed):
    tag=str(speed).replace('.','p')
    return Path(split)/m/f'seed{seed}_speed{tag}mps.csv'


def audit_episode(env,policy_class,config,split,m,seed,speed,tmp_file):
    errors=[];count=0;min_clearance=None;max_abs_lat=0.;max_abs_heading=0.
    mean_speed_acc=0.;max_speed=0.;raw_overrange=0;clean_candidate=0
    has_curve=False;near_edge_count=0;road_marked=0;last_info={}
    status='step_limit';prev_physical=None;prev_planned=None
    env.reset(seed=seed)
    policy=get_policy(env)
    if not isinstance(policy,policy_class):raise RuntimeError('Ego policy is not RecordedIDMPolicy')
    tmp_file.parent.mkdir(parents=True,exist_ok=True)
    with tmp_file.open('w',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=FIELDS,extrasaction='raise')
        writer.writeheader()
        try:
            for step in range(config['steps']):
                pre=state_before_action(env.agent)
                if pre.get('edge_clearance_proxy_m') is not None:
                    margin=pre['edge_clearance_proxy_m']
                    min_clearance=margin if min_clearance is None else min(margin,min_clearance)
                    if margin<config['edge_warning_m']:near_edge_count+=1
                max_abs_lat=max(max_abs_lat,abs(pre['planned_lateral_right_m']))
                max_abs_heading=max(max_abs_heading,abs(pre['heading_error_rad']))
                has_curve|=bool(pre['is_circular_lane'] or abs(pre['curvature_proxy_rad_per_m'])>1e-5)
                speed_actual=pre['speed_mps']
                mean_speed_acc+=speed_actual;max_speed=max(max_speed,speed_actual)
                before=policy.record_count
                physical_transition=(prev_physical is not None and pre['vehicle_lane_id']!=prev_physical)
                planned_transition=(prev_planned is not None and pre['planned_lane_id']!=prev_planned)
                _,_,terminated,truncated,info=env.step([0.0,0.0])
                last_info=info if isinstance(info,dict) else {}
                policy_after=get_policy(env)
                if policy_after is not policy or policy.record_count!=before+1:
                    raise RuntimeError('IDM policy changed or act() was not called exactly once')
                raw=checked_pair(policy.last_recorded_action,'policy.last_recorded_action')
                applied=read_applied_action(env.agent,raw)
                row=make_step_row(pre,raw,applied,split,m,seed,speed,step,last_info,terminated,truncated,
                                  physical_transition,planned_transition)
                row.update(post_state(env.agent))
                writer.writerow(row)
                count+=1;raw_overrange+=row['raw_out_of_range'];clean_candidate+=row['trainable_row']
                road_marked+=int(bool(row['on_yellow_continuous_line']))
                prev_physical=pre['vehicle_lane_id'];prev_planned=pre['planned_lane_id']
                if terminated or truncated:
                    if last_info.get('arrive_dest'):status='arrive_dest'
                    elif last_info.get('out_of_road'):status='out_of_road'
                    elif row['crash']:status='crash'
                    elif truncated:status='truncated'
                    else:status='other_termination'
                    break
        except Exception as e:
            errors.append(f'{type(e).__name__}: {e}')
            status='episode_error'
    success=status=='arrive_dest' and count>0 and not errors
    quality_reasons=[]
    if not success:quality_reasons.append(f'not_success:{status}')
    if count<config['min_episode_steps']:quality_reasons.append('episode_too_short')
    if max_abs_lat>config['max_lateral_m']:quality_reasons.append('lateral_error')
    if max_abs_heading>config['max_heading_rad']:quality_reasons.append('heading_error')
    if min_clearance is None or min_clearance<config['min_clearance_m']:quality_reasons.append('edge_margin')
    if near_edge_count>0:quality_reasons.append('near_edge')
    if road_marked>0:quality_reasons.append('road_marking')
    result=dict(split=split,map=m,seed=seed,speed_mps=speed,status=status,
                success=success,quality_pass=not quality_reasons,
                quality_reasons=quality_reasons,steps=count,errors=errors,
                mean_speed_mps=mean_speed_acc/max(count,1),max_speed_mps=max_speed,
                max_abs_lateral_m=max_abs_lat,max_abs_heading_error_rad=max_abs_heading,
                min_edge_clearance_proxy_m=min_clearance,near_edge_steps=near_edge_count,
                raw_out_of_range_steps=raw_overrange,
                candidate_trainable_steps=clean_candidate,actual_trainable_steps=0,
                is_curved_trajectory=bool(has_curve),
                recorded_actions_aligned=(count>0 and not errors),
                termination_flags={k:bool(last_info.get(k)) for k in ('arrive_dest','out_of_road','crash','on_yellow_continuous_line')},
                training_labels_source='MetaDrive vehicle.current_action (verified against actual vehicle controls)',
                raw_idm_label_used_directly=False)
    return result


def export_clean(audit_file,clean_file,episode):
    if not episode['quality_pass']:
        if clean_file.exists():clean_file.unlink()
        return 0
    clean_file.parent.mkdir(parents=True,exist_ok=True)
    tmp=clean_file.with_suffix('.tmp')
    kept=0
    with audit_file.open('r',newline='',encoding='utf-8') as src, tmp.open('w',newline='',encoding='utf-8') as dst:
        r=csv.DictReader(src)
        w=csv.DictWriter(dst,fieldnames=FIELDS);w.writeheader()
        for row in r:
            if row['trainable_row']=='1':
                w.writerow(row);kept+=1
    os.replace(tmp,clean_file)
    return kept


def new_manifest(conf):
    return {'version':VERSION,'config':conf,'config_sha256':stable_hash(conf),
            'warnings':[
                'MetaDrive simulation only: no real-world driving safety claim.',
                'Labels are vehicle.current_action, confirmed against vehicle controls and clipped IDM raw action.',
                'Any disagreement fails episode; raw policy commands are retained for audit.',
                'Only successful quality-approved episodes contribute to clean/',
                'Raw out-of-range IDM actions excluded from clean/ but remain in audit/.',
                'Geometry uses privileged MetaDrive lane map; this is not camera-only autonomy.',
                'Train/val/test are split by disjoint seeds, not random rows.',
                'MetaDrive road-edge clearance is an estimate, not a safety guarantee.',
                'No learned steering or longitudinal model is trained by this program.',
            ],'episodes':[],'totals':{}}


def refresh(manifest):
    eps=manifest['episodes']
    manifest['totals']={
        'episodes':len(eps),'success':sum(e['success'] for e in eps),
        'quality_accepted':sum(e['quality_pass'] for e in eps),
        'audit_steps':sum(e['steps'] for e in eps),
        'clean_steps':sum(e['actual_trainable_steps'] for e in eps),
        'raw_overrange_steps':sum(e['raw_out_of_range_steps'] for e in eps),
        'curved_episodes':sum(e['is_curved_trajectory'] for e in eps),
        'by_split':{s:{'episodes':sum(e['split']==s for e in eps),
                       'clean_steps':sum(e['actual_trainable_steps'] for e in eps if e['split']==s)}
                    for s in ('train','val','test')}
    }
    return manifest


def run(args,environment_factory=None,policy_class_factory=None):
    maps=parse_maps(args.maps);speeds=parse_speeds(args.speeds);splits=split_seed_lists(args)
    if not 10<=args.steps<=6000 or not 0<args.edge_warning_m<2 or not 1<=args.min_episode_steps<=args.steps:
        raise ValueError('Invalid steps/edge warning/min_episode_steps')
    if not (0.1<=args.max_lateral_m<=2. and 0.01<=args.max_heading_rad<=math.pi and
            0.<=args.min_clearance_m<=1.5):raise ValueError('Invalid quality thresholds')
    config=dict(maps=maps,speeds_mps=speeds,split_seeds=splits,steps=args.steps,
                edge_warning_m=args.edge_warning_m,max_lateral_m=args.max_lateral_m,
                max_heading_rad=args.max_heading_rad,min_clearance_m=args.min_clearance_m,
                min_episode_steps=args.min_episode_steps,physics_world_step_size=.02,
                decision_repeat=5,traffic_density=0.,metadrive_version_expected='0.4.3')
    root=Path(args.out_dir)
    index=root/'manifest.json'
    if index.exists():
        if not args.resume:raise FileExistsError(f'{index} exists. Use --resume or a new --out-dir.')
        manifest=json.loads(index.read_text(encoding='utf-8'))
        if manifest.get('version')!=VERSION or manifest.get('config_sha256')!=stable_hash(config):
            raise ValueError('Resume configuration mismatch: refusing to mix datasets')
    else:
        if args.resume and root.exists() and any(root.iterdir()):
            raise ValueError('Output exists but manifest missing: use new directory')
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Output directory not empty; choose new --out-dir')
        manifest=new_manifest(config);atom_json(index,manifest)
    completed={(e['split'],e['map'],e['seed'],e['speed_mps']) for e in manifest['episodes']}
    all_seeds=[x for xs in splits.values() for x in xs]
    if policy_class_factory is None:policy_class_factory=make_policy
    if environment_factory is None:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        environment_factory=MetaDriveEnv
    try:
        for m in maps:
            for speed in speeds:
                todo=[(split,seed) for split,seeds in splits.items() for seed in seeds
                      if (split,m,seed,speed) not in completed]
                if not todo:continue
                policy_class=policy_class_factory(speed*3.6)
                env=environment_factory(dict(map=m,use_render=False,
                    num_scenarios=max(all_seeds)+1,traffic_density=0.,
                    agent_policy=policy_class,log_level=40,
                    horizon=max(args.steps+10,1000),
                    physics_world_step_size=.02,decision_repeat=5))
                try:
                    for split,seed in todo:
                        relative=episode_rel_path(split,m,seed,speed)
                        audit_file=root/'audit'/relative
                        tmp_file=audit_file.with_suffix('.partial')
                        clean_file=root/'clean'/relative
                        # Stale uncommitted data can exist from interruption: discard it.
                        if tmp_file.exists():tmp_file.unlink()
                        result=audit_episode(env,policy_class,config,split,m,seed,speed,tmp_file)
                        audit_file.parent.mkdir(parents=True,exist_ok=True)
                        os.replace(tmp_file,audit_file)
                        result['audit_file']=str(Path('audit')/relative).replace('\\','/')
                        result['clean_file']=str(Path('clean')/relative).replace('\\','/') if result['quality_pass'] else None
                        result['actual_trainable_steps']=export_clean(audit_file,clean_file,result)
                        manifest['episodes'].append(result)
                        refresh(manifest);atom_json(index,manifest)
                        print(f'{m} {split} seed={seed} speed={speed:g} '
                              f'status={result["status"]} quality={result["quality_pass"]} '
                              f'clean={result["actual_trainable_steps"]}/{result["steps"]} '
                              f'out_of_range={result["raw_out_of_range_steps"]}',flush=True)
                finally:env.close()
    except KeyboardInterrupt:
        print('Interrupted. Completed episodes remain saved; rerun with --resume.',file=sys.stderr)
        raise
    refresh(manifest);atom_json(index,manifest)
    print('Saved manifest:',index.resolve())
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',default='S,C,X')
    p.add_argument('--speeds',default='2,3,4',help='Target speeds in m/s, e.g. 2,3,4')
    p.add_argument('--train-seeds',default='170-185')
    p.add_argument('--val-seeds',default='186-189')
    p.add_argument('--test-seeds',default='190-193')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--edge-warning-m',type=float,default=.35)
    p.add_argument('--max-lateral-m',type=float,default=.60)
    p.add_argument('--max-heading-rad',type=float,default=.12)
    p.add_argument('--min-clearance-m',type=float,default=.30)
    p.add_argument('--min-episode-steps',type=int,default=20)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v312b_idm_dataset'))
    p.add_argument('--resume',action='store_true')
    return run(p.parse_args(argv))


if __name__=='__main__':main()
