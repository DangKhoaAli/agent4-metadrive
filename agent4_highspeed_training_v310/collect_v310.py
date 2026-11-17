"""V3.10 collect MetaDrive ON-TEACHER-policy rule_lateral steering with pulses.

PDD V2 frozen; policy from V3.1 'lateral_3' (kp=.080,kh=.30) is the teacher.
Collected labels are PRE-PERTURBATION teacher steering at the SAME pre-step
observation; injected steering pulses create recovery states. Only completed
teacher trajectories enter training; failures remain in manifest. No IDM.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import joblib
import numpy as np

from common_v310 import (ROOT,DATA_STATUS,DEFAULT_SPEEDS,ALLOWED_MAPS,
    check_pdd,finite,clip,feature_names,state_features,
    validate_seed_groups,parse_seed_spec,parse_values,perturbation)

sys.path.insert(0,str(ROOT/'agent4_controller_v30'))
sys.path.insert(0,str(ROOT/'agent4_controller_v31'))
import ab_closed_loop_v30 as v30
from profiles_v31 import PROFILES,options_for
from residual_controller_v30 import steering_action

META=['map','seed','split','step','target_speed_mps','noise_md',
      'teacher_steer_md','applied_steer_md','pdd_steer_md','teacher_residual_md']


def collect_one(env,map_name,seed,speed,pdd,steps,amp,pulse_steps,pdd_names):
    """Teacher controls every step. Never accept states if step/observation failed."""
    opt=options_for(PROFILES['lateral_3'],steps=steps,trace_stride=20,tail_steps=35)
    env.reset(seed=seed)
    rows=[];status='step_limit';errors=[]
    prev=0.;n=0; n_perturb=0;n_recovery=0;max_lateral=0.;
    end_flags={}; distance=0.
    for t in range(steps):
        try:
            car=env.agent
            physical=v30.physical_road_state(env,car)
            if physical['predicted_out_of_road']:
                status='preexisting_out_of_road_guard';break
            local_lateral,_=v30.lane_offset(car)
            if local_lateral is not None and abs(local_lateral)>opt.guard_lane_offset_m:
                status='lane_departure_guard';break
            f,route=v30.feature_for_model(env,opt.max_gap)
            geom=v30.planned_geometry(car,opt.max_gap)
            e=finite(geom['lateral_right_m'],'lateral')
            h=finite(geom['heading_error_rad'],'heading')
            max_lateral=max(max_lateral,abs(e))
            if abs(h)>opt.guard_heading_rad:
                status='heading_departure_guard';break
            base=-float(v30.predict_steering(pdd,f))
            correction=steering_action(-base,'residual',e,h,opt.kp_lateral,
                      opt.kp_heading,opt.residual_limit,opt.steer_gain,
                      opt.steer_limit,prev,opt.max_steer_delta)
            teacher=correction['steer']
            long_action,_=v30.governed_speed_action(float(f['speed_mps']),float(speed),
                                                    float(route['remaining_m']))
            noise=perturbation(t,seed,amp,pulse_steps)
            executed=clip(teacher+noise,opt.steer_limit)
            features=state_features(f,pdd_names,base,e,h,route['adapted'])
            row={'map':map_name,'seed':seed,'split':'','step':t,
                 'target_speed_mps':float(speed),'noise_md':noise,
                 'teacher_steer_md':teacher,'applied_steer_md':executed,
                 'pdd_steer_md':base,'teacher_residual_md':teacher-base,
                 **features}
            before=np.asarray(car.position[:2],dtype=float)
            _,_,terminated,truncated,info=env.step([executed,float(long_action)])
            distance+=float(np.linalg.norm(np.asarray(env.agent.position[:2])-before))
            prev=float(executed)
            n+=1;rows.append(row)
            n_perturb+=int(noise!=0.)
            n_recovery+=int(abs(e)>=0.025 or abs(h)>=0.01)
            if terminated or truncated:
                end_flags={'arrive_dest':bool(info.get('arrive_dest',False)),
                           'crash':bool(info.get('crash',False)),
                           'out_of_road':bool(info.get('out_of_road',False)),
                           'route_completion':float(info.get('route_completion',0.))}
                status=('arrive_dest' if info.get('arrive_dest') else
                        'out_of_road' if info.get('out_of_road') else
                        'crash' if info.get('crash') else 'terminated_or_truncated')
                break
        except Exception as exc:
            status='episode_error';errors.append(f't={t} {type(exc).__name__}: {exc}')
            break
    s={'map':map_name,'seed':int(seed),'speed_mps':float(speed),'status':status,
       'n_applied_steps':n,'aligned_rows':len(rows),'noise_steps':n_perturb,
       'recovery_state_steps':n_recovery,'max_abs_lateral_m':max_lateral,
       'travelled_m':distance,'terminal_info':end_flags,'errors':errors,
       'rows_retained':len(rows) if status=='arrive_dest' and not errors else 0}
    return rows,s


def run(args):
    maps=parse_values(args.maps,ALLOWED_MAPS)
    speeds=parse_values(args.speeds,DEFAULT_SPEEDS,float)
    splits=validate_seed_groups(*(parse_seed_spec(x) for x in (
        args.train_seeds,args.val_seeds,args.eval_seeds)))
    if not 100<=args.steps<=5000:raise ValueError('steps must be 100..5000')
    perturbation(1,130,args.perturb_md,args.pulse_steps)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Output folder not empty; choose fresh --out-dir: {args.out_dir}')
    pdd=joblib.load(args.pdd)
    pdd_names=check_pdd(pdd)
    from metadrive.envs.metadrive_env import MetaDriveEnv
    args.out_dir.mkdir(parents=True,exist_ok=True)
    fields=META+feature_names(pdd_names)
    report={'status':DATA_STATUS,'maps':maps,'speeds_mps':speeds,
            'splits':splits,'perturb_md':args.perturb_md,'pulse_steps':args.pulse_steps,
            'steps_cap':args.steps,'teacher':'frozen PDD V2 + V3.1 rule_lateral (.08*e-.30*h clipped to .08); no IDM',
            'labels':'unperturbed teacher steering before env.step; noise only executed action',
            'data_fields':fields,'rows_by_split':{'train':0,'val':0},
            'rows_by_cell':{},'episodes':[],
            'limitations':['Research in MetaDrive only, no traffic, privileged geometry.',
                           'Only completed teacher episodes retained; failures preserved in manifest.',
                           'Low-noise teacher trajectories do NOT certify recovery from large drift.',
                           'Evaluation seeds are reserved and NEVER collected.']}
    with (args.out_dir/'train.csv').open('w',newline='',encoding='utf-8') as ftr,\
         (args.out_dir/'val.csv').open('w',newline='',encoding='utf-8') as fva:
        writers={k:csv.DictWriter(f,fieldnames=fields) for k,f in [('train',ftr),('val',fva)]}
        for w in writers.values():w.writeheader()
        for m in maps:
            env=MetaDriveEnv(dict(use_render=False,map=m,traffic_density=0.,
                num_scenarios=max(splits['train']+splits['val'])+1,
                horizon=max(1000,args.steps+10),log_level=40,
                physics_world_step_size=.02,decision_repeat=5))
            try:
                for speed in speeds:
                    for split in ('train','val'):
                        for seed in splits[split]:
                            rows,record=collect_one(env,m,seed,speed,pdd,args.steps,
                                args.perturb_md,args.pulse_steps,pdd_names)
                            record['split']=split
                            if record['rows_retained']:
                                for row in rows:row['split']=split
                                writers[split].writerows(rows)
                                report['rows_by_split'][split]+=len(rows)
                                key=f'{split}:{m}:{speed:g}'
                                report['rows_by_cell'][key]=report['rows_by_cell'].get(key,0)+len(rows)
                            report['episodes'].append(record)
                            print(f'{m:5s} {speed:4g}m/s {split:5s} seed={seed} '
                                f'{record["status"]:29s} rows={record["rows_retained"]:4d} '
                                f'perturb={record["noise_steps"]} recovery={record["recovery_state_steps"]}',
                                flush=True)
            finally:env.close()
    (args.out_dir/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,
            allow_nan=False),encoding='utf-8')
    print('Manifest:',(args.out_dir/'manifest.json').resolve())
    print('Rows:',report['rows_by_split'])
    print('IMPORTANT: inspect accepted coverage, perturb and recovery before training.')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',required=True,type=Path)
    p.add_argument('--maps',default='S,SSSSS')
    p.add_argument('--speeds',default='4,6,8,10')
    p.add_argument('--train-seeds',default='130-141')
    p.add_argument('--val-seeds',default='142-145')
    p.add_argument('--eval-seeds',default='146-149',help='Reserved: do not collect')
    p.add_argument('--steps',default=2500,type=int)
    p.add_argument('--perturb-md',default=.0035,type=float)
    p.add_argument('--pulse-steps',default=6,type=int)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v310_steering_data'))
    return run(p.parse_args(argv))

if __name__=='__main__':main()
