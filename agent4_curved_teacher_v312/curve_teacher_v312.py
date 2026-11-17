"""V3.12: MetaDrive IDMPolicy reference driver on C/X, geometry diagnostics.

No learner, no steering substitution and no new training. Uses the IDM policy
installed by MetaDrive as agent_policy, not the placeholder env.step action.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from geometry_v312 import compare_vehicle_route, approximate_vehicle_clearance, finite

VERSION = 'V312_IDM_CURVE_TEACHER_DIAGNOSTIC'
STEP_S = .1  # decision_repeat=5 and physics_world_step_size=.02
MAPS = ('C', 'X')
COLS = ('map','seed','policy','idm_speed_kmh','status','confirmed_success',
        'steps','elapsed_s','route_completion','max_speed_mps','mean_speed_mps',
        'max_abs_planned_lateral_m','max_abs_heading_error_rad',
        'min_estimated_edge_clearance_m','near_edge_warning_steps',
        'policy_action_count','recorded_actions_aligned','policy_capture_errors',
        'geometry_missing_steps','errors')


def seed_list(spec):
    result=[]
    for part in spec.split(','):
        item=part.strip()
        if not item:continue
        if '-' in item:
            nums=item.split('-')
            if len(nums)!=2 or not all(p.isdigit() for p in nums):
                raise ValueError('Use seeds e.g. 158,159 or 158-161')
            lo,hi=map(int,nums)
            if not lo<=hi or hi-lo>100:raise ValueError('Invalid seed range')
            result.extend(range(lo,hi+1))
        elif item.isdigit():result.append(int(item))
        else:raise ValueError('Invalid seeds')
    if not result or len(result)!=len(set(result)) or min(result)<0:
        raise ValueError('Seeds must be unique nonnegative integers')
    return result


def parse_maps(spec):
    names=[x.strip() for x in spec.split(',') if x.strip()]
    if not names or len(names)!=len(set(names)) or any(x not in MAPS for x in names):
        raise ValueError('Only C,X maps supported')
    return names


def get_ego_policy(env):
    for key in (getattr(env.agent,'id',None),getattr(env.agent,'name',None)):
        if key is None:continue
        try:
            p=env.engine.get_policy(key)
            if p is not None:return p
        except (TypeError, KeyError, AttributeError):
            continue
    raise RuntimeError('Cannot get ego IDM policy via env.engine.get_policy(ego.id/name)')


def make_recorded_policy(idm_speed_kmh):
    from metadrive.policy.idm_policy import IDMPolicy

    class RecordedIDMPolicy(IDMPolicy):
        # IDMPolicy.NORMAL_SPEED is a class attribute in km/h. None uses default.
        if idm_speed_kmh is not None:
            NORMAL_SPEED=float(idm_speed_kmh)

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.record_count=0
            self.last_recorded_action=None

        def act(self,*args,**kwargs):
            action=super().act(*args,**kwargs)
            a=[float(x) for x in action]
            if len(a)!=2 or not all(math.isfinite(x) for x in a):
                raise ValueError(f'IDMPolicy emitted invalid action: {action!r}')
            self.record_count+=1
            self.last_recorded_action=a
            return action
    return RecordedIDMPolicy


def validate_action_count(before, after, action):
    if after!=before+1:
        raise RuntimeError(f'IDM.act count mismatch {before} -> {after}; policy may not control ego')
    if not isinstance(action, (list,tuple)) or len(action)!=2:
        raise RuntimeError(f'Missing or malformed IDM action {action!r}')
    a=[float(v) for v in action]
    if not all(math.isfinite(v) for v in a):
        raise RuntimeError(f'IDM action nonfinite {a!r}')
    return a


def terminal_status(terminated,truncated,info):
    if info.get('arrive_dest'):return 'arrive_dest'
    if info.get('out_of_road'):return 'out_of_road'
    if info.get('crash') or any(info.get(k) for k in ('crash_vehicle','crash_object','crash_building','crash_sidewalk')):
        return 'crash'
    if truncated:return 'truncated'
    if terminated:return 'other_termination'
    return 'step_limit'


def read_snapshot(agent):
    try:
        velocity=finite(getattr(agent,'speed_km_h',None))
        if velocity is None:
            raise ValueError('Could not read agent.speed_km_h')
        geometry=compare_vehicle_route(agent)
        pose=geometry.get('planned') or geometry.get('physical') or {}
        return {
            'speed_mps':velocity/3.6,
            'xy':[float(x) for x in agent.position[:2]],
            'heading_rad':finite(getattr(agent,'heading_theta',None)),
            'geometry':geometry,
            'planned_lateral_right_m':pose.get('lateral_right_m'),
            'heading_error_rad':pose.get('heading_error_rad'),
            'distance_to_lane_end_m':pose.get('distance_to_lane_end_m'),
            'curvature_proxy_rad_per_m':pose.get('lane_curvature_proxy_rad_per_m'),
            'edge_clearance_proxy_m':approximate_vehicle_clearance(agent,pose.get('heading_error_rad',0) or 0),
        }
    except (TypeError,AttributeError,ValueError,IndexError) as ex:
        # Observation failures are recorded; never change control to hide them.
        return {'snapshot_error':f'{type(ex).__name__}: {ex}', 'geometry':{}}


def event_marks(pre,prev):
    tags=[]
    lateral=pre.get('planned_lateral_right_m')
    if lateral is not None:
        for threshold in (.15,.30,.50):
            old=abs(prev.get('planned_lateral_right_m') or 0.) if prev else 0.
            if abs(lateral)>=threshold>old:tags.append(f'lateral_crossed_{threshold:.2f}m')
    g=pre.get('geometry',{})
    oldg=prev.get('geometry',{}) if prev else {}
    if prev and g.get('vehicle_lane_id')!=oldg.get('vehicle_lane_id'):
        tags.append('physical_lane_transition')
    if prev and g.get('planned_lane_id')!=oldg.get('planned_lane_id'):
        tags.append('planned_lane_transition')
    remaining=pre.get('distance_to_lane_end_m')
    old_rem=prev.get('distance_to_lane_end_m') if prev else None
    if remaining is not None and remaining<=10 and (old_rem is None or old_rem>10):
        tags.append('within_10m_of_lane_end')
    if g.get('vehicle_lane_differs_from_plan'):tags.append('physical_plan_lane_mismatch')
    return tags


def empty_summary():
    return {'steps':0,'elapsed_s':0.,'route_completion':None,'max_speed_mps':None,
            'mean_speed_mps':None,'max_abs_planned_lateral_m':None,
            'max_abs_heading_error_rad':None,'min_estimated_edge_clearance_m':None,
            'near_edge_warning_steps':0,'policy_action_count':0,
            'recorded_actions_aligned':False,'policy_capture_errors':0,
            'geometry_missing_steps':0,'confirmed_success':False}


def evaluate_episode(env,recorded_policy_type,m,seed,steps,trace_stride,tail_size,near_edge_m):
    trace=[]; tail=deque(maxlen=tail_size)
    errors=[]; status='step_limit'; traveled=0.; speeds=[]
    lat_max=None;head_max=None;min_margin=None;route_done=None
    warn=0;geometry_missing=0;actions=0;policy_error_count=0
    prev=None;last_info={}
    try:
        env.reset(seed=seed)
        policy=get_ego_policy(env)
        if not isinstance(policy,recorded_policy_type):
            raise TypeError(f'Ego policy is {type(policy).__name__}, not RecordedIDMPolicy; placeholder control unsafe to interpret')
        for i in range(steps):
            pre=read_snapshot(env.agent)
            if pre.get('snapshot_error'):geometry_missing+=1
            elif not pre.get('geometry',{}).get('route_geometry_available'):
                geometry_missing+=1
            pose=pre.get('planned_lateral_right_m')
            head=pre.get('heading_error_rad')
            margin=pre.get('edge_clearance_proxy_m')
            if pose is not None:lat_max=max(abs(pose),lat_max or 0.)
            if head is not None:head_max=max(abs(head),head_max or 0.)
            if margin is not None:min_margin=margin if min_margin is None else min(min_margin,margin)
            if margin is not None and margin<near_edge_m:warn+=1
            if pre.get('speed_mps') is not None:speeds.append(pre['speed_mps'])
            before=get_ego_policy(env)
            if not isinstance(before,recorded_policy_type):
                raise TypeError('IDM policy was replaced while running')
            before_count=before.record_count
            pos=pre.get('xy')
            _,_,terminated,truncated,info=env.step([0.,0.])
            last_info=info if isinstance(info,dict) else {}
            try:
                after=get_ego_policy(env)
                if not isinstance(after,recorded_policy_type):
                    raise RuntimeError('IDM policy changed after env.step')
                action=validate_action_count(before_count,after.record_count,after.last_recorded_action)
            except Exception as ex:
                policy_error_count+=1
                raise RuntimeError(f'IDM control attribution failed at step {i}: {ex}') from ex
            actions+=1
            if pos is not None:
                newpos=[float(x) for x in env.agent.position[:2]]
                traveled+=math.hypot(newpos[0]-pos[0],newpos[1]-pos[1])
            route_done=finite(last_info.get('route_completion'))
            event={'step':i,'pre':pre,'idm_action':action,'route_completion_after':route_done,
                   'end_info_flags':{k:bool(last_info.get(k,False)) for k in ('arrive_dest','out_of_road','crash','crash_vehicle','crash_object','on_yellow_continuous_line')},
                   'terminated':bool(terminated),'truncated':bool(truncated)}
            event['events']=event_marks(pre,prev)
            tail.append(event)
            if (i%trace_stride==0 or event['events'] or terminated or truncated or
                (margin is not None and margin < near_edge_m)):
                trace.append(event)
            prev=pre
            if terminated or truncated:
                status=terminal_status(terminated,truncated,last_info)
                break
    except Exception as ex:
        status='episode_error' if actions==0 else 'policy_or_environment_error'
        errors.append(f'{type(ex).__name__}: {ex}')
    merged={x['step']:x for x in trace}
    merged.update({x['step']:x for x in tail})
    applied=actions
    confirmed=(status=='arrive_dest' and not errors and policy_error_count==0 and
               applied>0 and applied==len(speeds))
    summary=empty_summary()
    summary.update(steps=applied,elapsed_s=round(applied*STEP_S,5),
                   route_completion=route_done,travelled_approx_m=traveled,
                   max_speed_mps=max(speeds,default=None),
                   mean_speed_mps=sum(speeds)/len(speeds) if speeds else None,
                   max_abs_planned_lateral_m=lat_max,
                   max_abs_heading_error_rad=head_max,
                   min_estimated_edge_clearance_m=min_margin,
                   near_edge_warning_steps=warn,
                   policy_action_count=applied,
                   recorded_actions_aligned=(applied==len(speeds) and applied>0 and not errors),
                   policy_capture_errors=policy_error_count,
                   geometry_missing_steps=geometry_missing,
                   confirmed_success=confirmed,
                   used_idm_reference=True,trained_new_model=False)
    return {'map':m,'seed':seed,'profile':'idm_reference','status':status,
            'summary':summary,'errors':errors,
            'trace':sorted(merged.values(),key=lambda x:x['step']),
            'last_info_flags':{k:bool(last_info.get(k,False)) for k in ('arrive_dest','out_of_road','crash','crash_vehicle','crash_object','on_yellow_continuous_line')}}


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for b in iter(lambda:fh.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def summary_rows(episodes):
    groups=defaultdict(list)
    for e in episodes:groups[(e['map'],e['profile'])].append(e)
    return [dict(map=m,profile=p,total=len(rows),
                 confirmed_success=sum(bool(x['summary']['confirmed_success']) for x in rows),
                 route_termination_stats={s:sum(y['status']==s for y in rows)
                                          for s in sorted({x['status'] for x in rows})})
            for (m,p),rows in sorted(groups.items())]


def compare_prior(episodes,reference):
    """Comparison is outcome-only; speed distributions must be checked separately."""
    prior={(str(e.get('map')),int(e.get('seed')),str(e.get('profile'))):e for e in reference.get('episodes',[])}
    result=[]
    for e in episodes:
        for p in ('pdd_only','rule_lateral','learned_v310'):
            other=prior.get((e['map'],e['seed'],p))
            if other is None:continue
            old=other['summary']
            speed=finite(other.get('speed_mps'))
            result.append({'map':e['map'],'seed':e['seed'],'prior_profile':p,
                           'prior_target_speed_mps':speed,
                           'prior_status':old.get('status'),
                           'prior_confirmed_success':bool(old.get('confirmed_success')),
                           'idm_status':e['status'],
                           'idm_confirmed_success':bool(e['summary']['confirmed_success']),
                           'idm_mean_speed_mps':e['summary'].get('mean_speed_mps'),
                           'speed_matched_verified':False,
                           'warning':'Paired map/seed, not necessarily matched speed/state; compare only high-level outcomes.'})
    return result


def report_template(config,reference=None):
    return {'version':VERSION,'config':config,'no_new_model_trained':True,
            'controller':'MetaDrive IDMPolicy (instrumented subclass)',
            'simulator_only':True,
            'warnings':[
                'IDM is a third-party reference policy, not proof of safe driving.',
                'env.step([0,0]) is a placeholder; an actual IDM.act call must be recorded every step.',
                'NORMAL_SPEED km/h attribute is a target, NOT a hard speed cap. Check measured max/mean speed.',
                'No learned steering or learned longitudinal controls this test.',
                'The geometry/readouts and estimated edge margin are diagnostic proxies.',
                'MetaDrive termination flags remain authoritative; failed episodes must not be used as expert demos.',
                'Do not compare trajectory metrics causally across runs with different actual speeds.',
                'Traffic density is zero; no physical car claims.',
            ],'episodes':[],'group_scores':[],
            'prior_report_sha256':hash_file(reference) if reference is not None else None,
            'outcome_comparison':[]}


def refresh(report,reference_data=None):
    report['group_scores']=summary_rows(report['episodes'])
    report['outcome_comparison']=compare_prior(report['episodes'],reference_data) if reference_data else []
    return report


def atomic_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def export_csv(path,report):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=COLS);w.writeheader()
        for e in report['episodes']:
            s=e['summary'];row={'map':e['map'],'seed':e['seed'],'policy':e['profile'],
                                 'idm_speed_kmh':report['config']['idm_speed_kmh'],'status':e['status'],
                                 'errors':'; '.join(e['errors'][:2])}
            row.update({k:s.get(k) for k in COLS if k in s})
            w.writerow(row)
    tmp.replace(path)


def run(args):
    maps=parse_maps(args.maps)
    seeds=seed_list(args.seeds)
    if not (100<=args.steps<=6000 and 1<=args.trace_stride<=200 and
            5<=args.tail_steps<=200 and 0<=args.edge_warning_m<=2):
        raise ValueError('Invalid steps/stride/tail/edge warning options')
    if args.idm_speed_kmh is not None and (not math.isfinite(args.idm_speed_kmh) or
                                          not 2<=args.idm_speed_kmh<=60):
        raise ValueError('idm-speed-kmh must be in [2,60] or omitted for default')
    if args.resume and args.overwrite:
        raise ValueError('--resume and --overwrite are mutually exclusive')
    refpath=Path(args.reference) if args.reference else None
    if refpath is not None and not refpath.is_file():
        raise FileNotFoundError(refpath)
    reference=json.loads(refpath.read_text(encoding='utf-8')) if refpath else None
    conf=dict(maps=maps,seeds=seeds,steps=args.steps,trace_stride=args.trace_stride,
              tail_steps=args.tail_steps,edge_warning_m=args.edge_warning_m,
              idm_speed_kmh=args.idm_speed_kmh,physics_world_step_size=.02,
              decision_repeat=5,traffic_density=0.,reference_sha256=hash_file(refpath) if refpath else None)
    out=Path(args.out)
    if out.exists() and args.resume:
        report=json.loads(out.read_text(encoding='utf-8'))
        if report.get('version')!=VERSION or report.get('config')!=conf:
            raise ValueError('Report version/config differs; cannot resume')
        ids=[(x['map'],x['seed']) for x in report['episodes']]
        if len(ids)!=len(set(ids)):
            raise ValueError('Report contains duplicate episodes')
    else:
        if out.exists() and not args.overwrite:raise FileExistsError(f'{out} exists; use --resume or --overwrite')
        report=report_template(conf,refpath)
    completed={(x['map'],x['seed']) for x in report['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    policy_type=make_recorded_policy(args.idm_speed_kmh)
    try:
        for m in maps:
            todo=[seed for seed in seeds if (m,seed) not in completed]
            if not todo:continue
            env=MetaDriveEnv(dict(use_render=False,map=m,num_scenarios=max(seeds)+1,
                                  traffic_density=0.,agent_policy=policy_type,log_level=40,
                                  horizon=max(args.steps+10,1000),
                                  physics_world_step_size=.02,decision_repeat=5))
            try:
                for seed in todo:
                    e=evaluate_episode(env,policy_type,m,seed,args.steps,args.trace_stride,
                                       args.tail_steps,args.edge_warning_m)
                    report['episodes'].append(e)
                    refresh(report,reference)
                    atomic_write(out,report)
                    export_csv(out.with_suffix('.csv'),report)
                    s=e['summary']
                    print(f'{m} seed={seed} IDM {e["status"]} success={int(s["confirmed_success"])} '
                          f'actions={s["policy_action_count"]} max_speed={s["max_speed_mps"]} '
                          f'route={s["route_completion"]}',flush=True)
            finally:
                env.close()
    except KeyboardInterrupt:
        print('Stopped; finished episodes were saved. Use --resume to continue.',file=sys.stderr)
        raise
    refresh(report,reference)
    atomic_write(out,report);export_csv(out.with_suffix('.csv'),report)
    print(f'V3.12 saved {len(report["episodes"])} episodes to {out.resolve()}')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',default='C,X')
    p.add_argument('--seeds',default='158,159')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--trace-stride',type=int,default=15)
    p.add_argument('--tail-steps',type=int,default=55)
    p.add_argument('--edge-warning-m',type=float,default=.35)
    p.add_argument('--idm-speed-kmh',type=float,default=10.8,
                   help='IDM desired speed in km/h (10.8 = 3 m/s); not an enforced speed cap')
    p.add_argument('--reference',type=Path,default=None,
                   help='Optional V3.11B rescored JSON for map/seed-matched high-level outcomes')
    p.add_argument('--out',type=Path,default=Path('agent4_v312_idm_curves.json'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    return run(p.parse_args(argv))


if __name__=='__main__':
    main()
