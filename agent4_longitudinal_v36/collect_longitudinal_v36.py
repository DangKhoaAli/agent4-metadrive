"""V3.6: collect *EXECUTED* native IDM throttle/brake, unlike V3.5.

The IDM's NORMAL_SPEED / target_speed is changed before every .act to permit
speed transitions and final route slow-down. Its raw acceleration is bounded to
MetaDrive's [-1,1] actuator interval and actually applied to the vehicle.
No governor action is ever substituted for the IDM longitudinal output.
Only successful teacher episodes become training rows. Zero traffic, simulator.
"""
from __future__ import annotations
import argparse,csv,json,sys,math
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent.parent
for d,f in [('agent4_controller_v30','ab_closed_loop_v30.py'),
            ('agent4_calibration_v26','probe_idm_action_v26.py')]:
    if not (ROOT/d/f).exists():raise RuntimeError(f'Missing sibling {d}/{f}')
    sys.path.insert(0,str(ROOT/d))
import ab_closed_loop_v30 as v30
from probe_idm_action_v26 import get_policy
from common_v36 import (names,make_row,speed_schedule,check_splits,parse_seeds,
                         parse_speeds,parse_simple_csv,MAPS,status_from,finite)

CSV_META=['map','seed','step','base_speed_mps','lane_class',
          'teacher_steer_md','teacher_longitudinal','native_idm_longitudinal',
          'applied_longitudinal','longitudinal_clipped']


def make_recorded_idm_policy():
    from metadrive.policy.idm_policy import IDMPolicy
    class NativeSpeedIDMPolicy(IDMPolicy):
        """Instance-level desired speed; no override of native IDM acceleration."""
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.requested_speed_kmh=None
            self.last_record=None
            self.record_count=0
        def set_desired_speed(self, speed_mps):
            kmh=finite(speed_mps,'desired speed')*3.6
            if not 0.36<=kmh<=19.8:raise ValueError(f'Invalid desired speed {kmh} km/h')
            # IDMPolicy.lane_change_policy may reset target_speed to NORMAL_SPEED.
            self.NORMAL_SPEED=kmh
            self.CREEP_SPEED=min(kmh,5.0)
            self.target_speed=kmh
            self.requested_speed_kmh=kmh
        def act(self,*args,**kwargs):
            if self.requested_speed_kmh is None:
                raise RuntimeError('IDM target not set before simulator step')
            raw=super().act(*args,**kwargs)
            if len(raw)!=2:raise RuntimeError(f'IDM action must have two components: {raw}')
            native_steer,native_long=map(finite,raw)
            applied_steer=float(max(-.35,min(.35,native_steer)))
            applied_long=float(max(-1.,min(1.,native_long)))
            self.last_record={'native_steer':native_steer,
                              'native_longitudinal':native_long,
                              'teacher_steer_md':applied_steer,
                              'teacher_longitudinal':applied_long,
                              'applied_longitudinal':applied_long,
                              'longitudinal_clipped':int(abs(native_long-applied_long)>1e-8),
                              'desired_speed_kmh':self.requested_speed_kmh,
                              'idm_target_after_act_kmh':finite(self.target_speed)}
            self.record_count+=1
            # Actual applied action = supervised target; NO governor override.
            return [applied_steer,applied_long]
    return NativeSpeedIDMPolicy


def collect_episode(env,map_name,seed,base_speed,steps,pdd_names,enable_schedule):
    env.reset(seed=seed)
    policy=get_policy(env)
    if not hasattr(policy,'set_desired_speed'):
        raise RuntimeError(f'Expected NativeSpeedIDMPolicy; got {type(policy).__name__}')
    kept=[]; errors=[]; status='step_limit'; last_info={}
    stats=Counter(); min_speed=math.inf;max_speed=0.0
    for step in range(steps):
        car=env.agent
        lane_class_before=type(car.lane).__name__ if car.lane is not None else 'None'
        requested=speed_schedule(base_speed,step,enable_schedule)
        row=None
        try:
            features,route=v30.feature_for_model(env,2.0)
            geom=v30.planned_geometry(car,2.0)
            # Route planner only supplies a desired speed; IDM selects actual action.
            _,route_target=v30.governed_speed_action(float(features['speed_mps']),
                                                      requested,float(route['remaining_m']))
            desired=max(0.1,float(route_target))
            row=make_row(features,pdd_names,geom,route['adapted'],desired,route['remaining_m'])
        except Exception as exc:
            # Fail visibly and discard this episode: cannot trust time alignment.
            errors.append(f'step {step} observation: {type(exc).__name__}: {exc}')
            status='observation_error';break
        policy.set_desired_speed(desired)
        pre_count=policy.record_count
        try:
            _,_,terminated,truncated,info=env.step([0.,0.])
        except Exception as exc:
            errors.append(f'step {step} env: {type(exc).__name__}: {exc}')
            status='env_step_error';break
        actual=get_policy(env)
        if actual.record_count!=pre_count+1 or actual.last_record is None:
            errors.append(f'step {step} teacher/action misalignment')
            status='alignment_error';break
        a=actual.last_record
        if abs(a['teacher_longitudinal']-a['applied_longitudinal'])>1e-12:
            errors.append(f'step {step} label not equal to executed action')
            status='alignment_error';break
        item={'map':map_name,'seed':seed,'step':step,
              'base_speed_mps':base_speed,
              'lane_class':lane_class_before,
              'teacher_steer_md':a['teacher_steer_md'],
              'teacher_longitudinal':a['teacher_longitudinal'],
              'native_idm_longitudinal':a['native_longitudinal'],
              'applied_longitudinal':a['applied_longitudinal'],
              'longitudinal_clipped':a['longitudinal_clipped']}
        item.update(row)
        kept.append(item)
        stats['actions']+=1
        stats['brake_actions']+=int(a['teacher_longitudinal']<-.02)
        stats['positive_actions']+=int(a['teacher_longitudinal']>.02)
        stats['zero_actions']+=int(abs(a['teacher_longitudinal'])<=.02)
        stats['longitudinal_clipped']+=a['longitudinal_clipped']
        stats['target_reduced_steps']+=int(desired < base_speed-.05)
        min_speed=min(min_speed,float(features['speed_mps']))
        max_speed=max(max_speed,float(features['speed_mps']))
        if terminated or truncated:
            last_info={'arrive_dest':bool(info.get('arrive_dest',False)),
                       'out_of_road':bool(info.get('out_of_road',False)),
                       'crash':bool(info.get('crash',False)),
                       'route_completion':float(info.get('route_completion',0))}
            status=status_from(last_info,terminated,truncated)
            break
    summary={'map':map_name,'seed':seed,'base_speed_mps':base_speed,
             'status':status,'n_steps':stats['actions'],
             'stats':dict(stats),'min_observed_speed_mps':None if not kept else min_speed,
             'max_observed_speed_mps':max_speed,
             'info':last_info,'sample_errors':errors[:10],
             'rows_accepted_for_training':len(kept) if status=='arrive_dest' else 0}
    return kept,summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',default='S,C,X')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--train-seeds',default='60-71')
    p.add_argument('--val-seeds',default='72-75')
    p.add_argument('--eval-seeds',default='76-79',help='NEVER collected')
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v36_idm_data'))
    p.add_argument('--no-speed-schedule',action='store_true',
                   help='Disable speed transitions (not recommended: no deceleration coverage)')
    a=p.parse_args(argv)
    try:
        maps=parse_simple_csv(a.maps,MAPS)
        speeds=parse_speeds(a.speeds)
        groups=check_splits(parse_seeds(a.train_seeds),parse_seeds(a.val_seeds),parse_seeds(a.eval_seeds))
        if not 100<=a.steps<=5000:raise ValueError('steps must be 100..5000')
        if a.out_dir.exists() and any(a.out_dir.iterdir()):
            raise ValueError('Output directory not empty; choose a new --out-dir')
    except ValueError as exc:p.error(str(exc))
    from metadrive.envs.metadrive_env import MetaDriveEnv
    fields=CSV_META+names(v30.FEATURES)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'status':'V36_NATIVE_IDM_LONGITUDINAL_TEACHER_DATA',
              'data_contract':'teacher_longitudinal == applied_longitudinal (clipped IDM.act output)',
              'maps':maps,'speeds_mps':speeds,'splits':groups,
              'speed_schedule_enabled':not a.no_speed_schedule,
              'traffic_density':0.0,'data_fields':fields,
              'only_successful_teacher_episodes_retained':True,
              'rows_by_split':{'train':0,'val':0},
              'rows_by_speed_map':{},'episodes':[],
              'limitations':['IDM steering drives during collection; learned V3.5 steering drives during evaluation.',
                             'Target speed and near-destination slowdown are hand-planned; low-level throttle/brake is native IDM.',
                             'No traffic, so IDM following/yielding behavior is not learned.',
                             'Only successful teacher episodes are retained.']}
    with (a.out_dir/'train.csv').open('w',encoding='utf-8',newline='') as ftr, (a.out_dir/'val.csv').open('w',encoding='utf-8',newline='') as fva:
        writers={'train':csv.DictWriter(ftr,fieldnames=fields),
                 'val':csv.DictWriter(fva,fieldnames=fields)}
        for w in writers.values():w.writeheader()
        for m in maps:
            env=None
            try:
                env=MetaDriveEnv(dict(use_render=False,map=m,
                           num_scenarios=max(groups['train']+groups['val'])+1,
                           traffic_density=0.0,agent_policy=make_recorded_idm_policy(),
                           horizon=max(1000,a.steps+10),log_level=40))
                for speed in speeds:
                    for split in ('train','val'):
                        for seed in groups[split]:
                            try:
                                rows,episode=collect_episode(env,m,seed,speed,a.steps,
                                      v30.FEATURES,not a.no_speed_schedule)
                                if episode['status']=='arrive_dest':
                                    writers[split].writerows(rows)
                                    manifest['rows_by_split'][split]+=len(rows)
                                    key=f'{split}:{speed:g}:{m}'
                                    manifest['rows_by_speed_map'][key]=manifest['rows_by_speed_map'].get(key,0)+len(rows)
                            except Exception as exc:
                                episode={'map':m,'seed':seed,'base_speed_mps':speed,
                                         'status':'collection_error',
                                         'sample_errors':[f'{type(exc).__name__}: {exc}'],
                                         'rows_accepted_for_training':0,'stats':{}}
                            episode['split']=split
                            manifest['episodes'].append(episode)
                            print(f'{m} speed={speed:g} {split} seed={seed}: {episode["status"]} rows={episode["rows_accepted_for_training"]}',flush=True)
            finally:
                if env is not None:env.close()
    manifest['successes_by_split']={split:sum(e['split']==split and e['status']=='arrive_dest' for e in manifest['episodes']) for split in ('train','val')}
    manifest['total_brake_rows_retained']=sum(e['stats'].get('brake_actions',0) for e in manifest['episodes'] if e['status']=='arrive_dest')
    output=a.out_dir/'manifest.json'
    output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Manifest:',output.resolve(),'rows:',manifest['rows_by_split'],
          'brake rows (successes):',manifest['total_brake_rows_retained'])
    if min(manifest['rows_by_split'].values())<500:
        raise RuntimeError('Not enough teacher samples; inspect manifest')
    if manifest['total_brake_rows_retained']<30:
        raise RuntimeError('Insufficient brake labels; vary speed schedule or inspect IDM behavior')

if __name__=='__main__':main()
