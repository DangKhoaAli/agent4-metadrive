"""V3.5 closed-loop benchmark: frozen V3.3 student vs newly trained V3.5.

Tests 2/3/4 m/s with identical maps and seeds. No IDM during evaluation.
V3.4's actuator, governor and termination logic remain unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_robustness_v34'))
from benchmark_v34 import (parse_seeds,parse_speeds,parse_simple_csv,
                           eval_one,summary_rows,group_scores,v30,v33)

PROFILES=('student_v33','student_v35','lateral_3')


def check_student(student,pdd_features,seeds,expected_version):
    if student.get('version')!=expected_version:
        raise ValueError(f'Expected {expected_version}, received {student.get("version")}')
    if list(student.get('feature_order',[]))!=v33.feature_names(pdd_features):
        raise ValueError('Student feature order mismatch')
    used={int(z) for _,z in student.get('train_seed_groups',[])} | {int(z) for _,z in student.get('val_seed_groups',[])}
    if set(seeds)&used:
        raise ValueError(f'Closed-loop seed leakage: {set(seeds)&used}')
    if expected_version.startswith('V35_') and set(seeds)&set(range(40)):
        raise ValueError('V3.5 eval requires never-before-used seeds >=40')
    if not hasattr(student.get('student_model'), 'predict'):
        raise ValueError('Student estimator missing predict')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--old-student',type=Path,required=True)
    p.add_argument('--student',type=Path,required=True)
    p.add_argument('--maps',default='C,X')
    p.add_argument('--seeds',default='56,57')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--profiles',default='student_v33,student_v35')
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--trace-stride',type=int,default=30)
    p.add_argument('--out',type=Path,default=Path('agent4_v35_closed_loop_eval.json'))
    a=p.parse_args(argv)
    try:
        maps=parse_simple_csv(a.maps,('S','C','X'))
        speeds=parse_speeds(a.speeds)
        seeds=parse_seeds(a.seeds)
        profiles=parse_simple_csv(a.profiles,PROFILES)
        if a.steps<100 or not 1<=a.trace_stride<=100:
            raise ValueError('Bad steps or trace-stride')
    except ValueError as exc:p.error(str(exc))
    pdd=joblib.load(a.model)
    old_student=joblib.load(a.old_student)
    student=joblib.load(a.student)
    if list(pdd.get('features',[]))!=list(v30.FEATURES) or pdd.get('label_mode')!='at_t':
        raise ValueError('PDD base bundle incompatible')
    check_student(old_student,v30.FEATURES,seeds,'V33_META_DRIVE_IDM_DISTILLED_RESIDUAL')
    check_student(student,v30.FEATURES,seeds,'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report={'status':'V35_MULTISPEED_CLOSED_LOOP_NO_IDM',
            'maps':maps,'seeds':seeds,'speeds_mps':speeds,'profiles':profiles,
            'traffic_density':0.,'idm_used_for_control':False,'learned_longitudinal_used':False,
            'pdd_base_model_frozen':True,
            'seeds_never_used_for_training_or_tuning':True,
            'warnings':['Only arrive_dest counts as success.',
                        'The longitudinal speed governor is rule-based.',
                        'Cumulative road offset thresholds are only diagnostics.',
                        'No traffic or physical vehicle safety claims.'],
            'episodes':[]}
    for map_name in maps:
        env=None
        try:
            env=MetaDriveEnv(dict(use_render=False,map=map_name,
                                  num_scenarios=max(seeds)+1,traffic_density=0.,
                                  horizon=max(a.steps+10,1000),log_level=40))
            for seed in seeds:
                for speed in speeds:
                    for profile in profiles:
                        bundle=old_student if profile=='student_v33' else student
                        # V3.4 expects 'student_v33' to execute the learned student path.
                        internal='lateral_3' if profile=='lateral_3' else 'student_v33'
                        item=eval_one(env,map_name,seed,speed,internal,pdd,bundle,a.steps,a.trace_stride)
                        item['profile']=profile
                        report['episodes'].append(item)
                        s=item['summary']
                        print(f'{map_name} seed={seed} speed={speed:g} {profile} '
                              f'{s["status"]} {s["travelled_approx_m"]:.1f}m',flush=True)
        finally:
            if env is not None:env.close()
    rows=summary_rows(report['episodes'])
    report['scores_by_speed_map_profile']=group_scores(rows)
    report['success_by_profile']={name:{'success':sum(e['profile']==name and e['summary']['status']=='arrive_dest' for e in report['episodes']),
                                        'total':sum(e['profile']==name for e in report['episodes'])} for name in profiles}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Output:',a.out.resolve())
    print('Success:',report['success_by_profile'])


if __name__=='__main__':main()
