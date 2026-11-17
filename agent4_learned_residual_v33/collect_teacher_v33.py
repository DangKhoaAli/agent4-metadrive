"""V3.3 collect a NEW MetaDrive IDM steering-training dataset at 3 m/s.

MetaDrive IDM computes teacher STEERING at state t. A speed governor replaces
only the teacher longitudinal action. Optional bounded steering perturbations
on the *executed* action expose recovery states; the unperturbed teacher
steering remains the label. Only completed teacher episodes are included.

This is a second simulator teacher dataset, NOT CARLA PDD labels, and it is not
an evaluation of the learned model. Do not use on a physical vehicle.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
for folder, filename in [('agent4_controller_v30','ab_closed_loop_v30.py'),
                         ('agent4_calibration_v26','probe_idm_action_v26.py')]:
    directory=ROOT/folder
    if not (directory/filename).is_file():
        raise RuntimeError(f'Missing sibling dependency {directory/filename}')
    sys.path.insert(0,str(directory))
from common_v33 import (MAX_STEER,finite,clip,state_features,feature_names,seed_sets)
import ab_closed_loop_v30 as v30
from probe_idm_action_v26 import get_policy


def seeds(value):
    """Accept 4-23 or 4,5,8-10."""
    out=[]
    for chunk in value.split(','):
        s=chunk.strip()
        if not s: continue
        if '-' in s:
            a,b=map(int,s.split('-',1))
            if a>b: raise ValueError('Invalid seed interval')
            out.extend(range(a,b+1))
        else: out.append(int(s))
    return out


def steering_perturbation(step, seed, amplitude):
    """Deterministic signed ~1.2s pulses, no simulator random calls.

    Only influences the trajectory, NOT the teacher supervision target.
    """
    amp = finite(amplitude,'perturbation')
    if not 0 <= amp <= .035:
        raise ValueError('Perturbation magnitude must be in [0,0.035]')
    phase=(int(step)+int(seed)*29)%180
    if 45 <= phase < 57: return amp
    if 115 <= phase < 127: return -amp
    return 0.0


def csv_fields(pdd_names):
    return (['map','seed','step','lane_class','is_circular_lane','perturb_md',
             'teacher_steer_md','teacher_native_steer_md','applied_steer_md',
             'original_idm_longitudinal','governed_longitudinal']
            +feature_names(pdd_names))


def collect_episode(env,map_name,seed,bundle,pdd_names,steps,speed,perturb):
    from metadrive.policy.idm_policy import IDMPolicy
    # Actual injected policy class defined globally by make_slow_idm_policy().
    accepted=[]; row_errors=[]
    count=0; clipped_native=0; perturb_applied=0; duration=0
    last_info=None; status='step_limit'
    env.reset(seed=seed)
    for step in range(steps):
        policy = get_policy(env)
        if not isinstance(policy,IDMPolicy) or not hasattr(policy,'queue_action'):
            raise RuntimeError(f'SlowIDMPolicy unavailable for episode {map_name}/{seed}: {type(policy).__name__}')
        before_count=policy.record_count
        car=env.agent
        lane_class_before=type(car.lane).__name__ if car.lane is not None else 'None'
        try:
            f, route=v30.feature_for_model(env,2.0)
            geom=v30.planned_geometry(car,2.0)
            base=-float(v30.predict_steering(bundle,f))
            data=state_features(f,pdd_names,base,geom['lateral_right_m'],
                                geom['heading_error_rad'],route['adapted'])
            ok=True
        except Exception as exc:
            ok=False
            data={};route={'remaining_m':999.0}
            if len(row_errors)<12:row_errors.append(f'{step}:feature:{type(exc).__name__}:{exc}')
        raw_long, _ = v30.governed_speed_action(float(car.speed_km_h)/3.6,
                                                speed,route['remaining_m'])
        noise=steering_perturbation(step,seed,perturb)
        policy.queue_action(float(raw_long),noise)
        try:
            _, _, terminated,truncated,info=env.step([0.0,0.0])
        except Exception as exc:
            raise RuntimeError(f'env.step failure at step {step}: {type(exc).__name__}: {exc}') from exc
        policy_after=get_policy(env)
        if policy_after.record_count != before_count+1:
            raise RuntimeError(f'IDM action not time-aligned at {step}: {before_count}->{policy_after.record_count}')
        act=policy_after.last_record
        if act is None:
            raise RuntimeError(f'No recorded policy act at step {step}')
        duration+=1
        perturb_applied+=int(abs(noise)>0)
        clipped_native+=int(abs(act['native_steer'])>MAX_STEER+1e-9)
        if ok:
            row={'map':map_name,'seed':seed,'step':step,
                 'lane_class':lane_class_before,
                 'is_circular_lane':int('circular' in lane_class_before.lower()),
                 'perturb_md':noise,
                 'teacher_steer_md':act['teacher_steer'],
                 'teacher_native_steer_md':act['native_steer'],
                 'applied_steer_md':act['applied_steer'],
                 'original_idm_longitudinal':act['native_longitudinal'],
                 'governed_longitudinal':act['applied_longitudinal']}
            row.update(data)
            accepted.append(row)
        if terminated or truncated:
            last_info={'arrive_dest':bool(info.get('arrive_dest',False)),
                       'out_of_road':bool(info.get('out_of_road',False)),
                       'crash':bool(info.get('crash',False)),
                       'route_completion':float(info.get('route_completion',0))}
            if last_info['arrive_dest']:status='arrive_dest'
            elif last_info['out_of_road']:status='out_of_road'
            elif last_info['crash']:status='crash'
            else:status='terminated_or_truncated'
            break
    return accepted,{'map':map_name,'seed':seed,'status':status,
                     'n_steps':duration,'n_aligned_feature_rows':len(accepted),
                     'n_native_steer_clipped':clipped_native,'n_perturb_steps':perturb_applied,
                     'info':last_info,'sample_errors':row_errors,
                     'rows_accepted_for_training':len(accepted) if status=='arrive_dest' else 0}


def make_slow_idm_policy():
    from metadrive.policy.idm_policy import IDMPolicy
    class SlowRecordedIDMPolicy(IDMPolicy):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.queued_longitudinal=None
            self.queued_perturb=0.0
            self.record_count=0
            self.last_record=None
        def queue_action(self, governed_longitudinal, steering_perturbation):
            self.queued_longitudinal=finite(governed_longitudinal)
            self.queued_perturb=finite(steering_perturbation)
        def act(self,*args,**kwargs):
            raw=super().act(*args,**kwargs)
            if self.queued_longitudinal is None:
                raise RuntimeError('SlowIDMPolicy act before pre-step longitudinal set')
            if len(raw)!=2:
                raise RuntimeError(f'Unexpected IDM action: {raw}')
            raw_steer,raw_long=map(finite,raw)
            teacher=clip(raw_steer,MAX_STEER)
            applied=clip(teacher+self.queued_perturb,MAX_STEER)
            self.last_record={'native_steer':raw_steer,
                              'native_longitudinal':raw_long,
                              'teacher_steer':teacher,
                              'applied_steer':applied,
                              'applied_longitudinal':self.queued_longitudinal}
            self.record_count+=1
            return [applied,self.queued_longitudinal]
    return SlowRecordedIDMPolicy


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--maps',default='S,C,X')
    p.add_argument('--train-seeds',default='4-23')
    p.add_argument('--val-seeds',default='24-27')
    p.add_argument('--eval-seeds',default='28-31',help='reserved; NEVER collected')
    p.add_argument('--steps',type=int,default=1400)
    p.add_argument('--speed-mps',type=float,default=3.0)
    p.add_argument('--perturb-md',type=float,default=0.012)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v33_teacher_data'))
    a=p.parse_args(argv)
    try:
        a.maps=[v.strip() for v in a.maps.split(',') if v.strip()]
        if not a.maps or any(v not in ('S','C','X') for v in a.maps) or len(set(a.maps))!=len(a.maps):
            raise ValueError('Maps must be unique S,C,X')
        a.split=seed_sets(seeds(a.train_seeds),seeds(a.val_seeds),seeds(a.eval_seeds))
        if a.steps<50 or not 0.5<=a.speed_mps<=5:
            raise ValueError('Invalid steps or low-speed target')
        steering_perturbation(0,0,a.perturb_md)
    except ValueError as exc:p.error(str(exc))
    return a


def main():
    a=parse_args()
    pdd=joblib.load(a.model)
    if list(pdd.get('features',[]))!=list(v30.FEATURES) or pdd.get('label_mode')!='at_t':
        raise ValueError('Wrong PDD V2 bundle or feature order')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    a.out_dir.mkdir(parents=True,exist_ok=True)
    filenames={part:a.out_dir/f'{part}.csv' for part in ('train','val')}
    fields=csv_fields(v30.FEATURES)
    report={'status':'V33_SLOW_IDM_TEACHER_DATA_NOT_LEARNED_AGENT',
            'maps':a.maps,'splits':a.split,'model_path':str(a.model),
            'speed_target_mps':a.speed_mps,'perturbation':a.perturb_md,
            'data_fields':fields,'only_successful_episodes_kept':True,
            'simulator_policy':'SlowRecordedIDMPolicy: IDM steering + independent governor longitudinal',
            'episodes':[],'rows_by_split':{'train':0,'val':0},
            'limitations':['PDD V2 steering remains frozen and is a student input, not the teacher label.',
                           'Teacher labels come from MetaDrive IDM, NOT CARLA/PDD.',
                           'Steering teacher is bounded to [-0.35,0.35]; longitudinal teacher is replaced.',
                           'Perturbation affects executed action only; labels are unperturbed IDM steer.',
                           'Data from successes only can bias toward easy routes.',
                           'All observations are simulator privileged; no traffic or physical vehicle.',
                           'Do NOT use reserved eval seeds for training/validation.']}
    with filenames['train'].open('w',newline='',encoding='utf-8') as ftr, filenames['val'].open('w',newline='',encoding='utf-8') as fval:
        writers={'train':csv.DictWriter(ftr,fieldnames=fields),
                 'val':csv.DictWriter(fval,fieldnames=fields)}
        for w in writers.values():w.writeheader()
        for name in a.maps:
            env=None
            try:
                env=MetaDriveEnv(dict(use_render=False,map=name,
                        num_scenarios=max(a.split['train']+a.split['val'])+1,
                        traffic_density=0.0,agent_policy=make_slow_idm_policy(),
                        horizon=max(a.steps+10,1000),log_level=40))
                for split in ('train','val'):
                    for seed in a.split[split]:
                        try:
                            rows,s=collect_episode(env,name,seed,pdd,v30.FEATURES,a.steps,
                                                   a.speed_mps,a.perturb_md)
                            if s['status']=='arrive_dest':
                                writers[split].writerows(rows)
                                report['rows_by_split'][split]+=len(rows)
                            s['split']=split
                        except Exception as exc:
                            s={'map':name,'seed':seed,'split':split,
                               'status':'collection_error',
                               'error':f'{type(exc).__name__}: {exc}',
                               'rows_accepted_for_training':0}
                        report['episodes'].append(s)
                        print(f'{name} seed={seed} {split} {s["status"]} '
                              f'rows={s["rows_accepted_for_training"]}')
            finally:
                if env is not None: env.close()
    report['successes_by_split']={split:sum(1 for x in report['episodes'] if x.get('split')==split and x.get('status')=='arrive_dest') for split in ('train','val')}
    path=a.out_dir/'manifest.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved:',path.resolve(),' | Rows:',report['rows_by_split'])
    if min(report['rows_by_split'].values())<200:
        raise RuntimeError('Insufficient aligned teacher rows: inspect manifest.json before training')

if __name__=='__main__': main()
