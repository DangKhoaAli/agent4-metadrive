"""Closed-loop A/B: frozen PDD baseline, V3.1 lateral_3, learned V3.3 steering.

No IDM used during evaluation. Student steering = frozen PDD model steering
+ bounded *learned* residual, WITHOUT rule-based lateral/heading correction.
All modes use exactly the V3.1 speed governor, map settings and guards.
Default seeds 28,29 must be fully unseen in teacher train/validation CSV.
Simulation only, NEVER physical vehicles.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT=Path(__file__).resolve().parent.parent
for dirname,filename in [('agent4_controller_v30','ab_closed_loop_v30.py'),
                          ('agent4_controller_v31','profiles_v31.py')]:
    directory=ROOT/dirname
    if not (directory/filename).is_file():
        raise RuntimeError(f'Missing dependency {directory/filename}')
    sys.path.insert(0,str(directory))
import ab_closed_loop_v30 as v30
from profiles_v31 import PROFILES,options_for,episode_summary
from common_v33 import feature_names,state_features,student_steer_md

CHOICES=('baseline_3','lateral_3','student_v33')


def checked_student_bundle(student,pdd_features,evaluation_seeds):
    if student.get('version')!='V33_META_DRIVE_IDM_DISTILLED_RESIDUAL':
        raise ValueError('Not a V3.3 student bundle')
    if list(student.get('feature_order',[]))!=feature_names(pdd_features):
        raise ValueError('V3.3 model features do not match the current PDD bundle')
    training={int(seed) for _,seed in student.get('train_seed_groups',[])}
    validation={int(seed) for _,seed in student.get('val_seed_groups',[])}
    overlap=set(evaluation_seeds)&(training|validation)
    if overlap:
        raise ValueError(f'Closed-loop eval seeds leaked into teacher data: {sorted(overlap)}')
    return True


def pilot_student_episode(env,seed,pdd_bundle,student,*,steps=1400,trace_stride=10):
    opts=options_for(PROFILES['lateral_3'],steps=steps,trace_stride=trace_stride,tail_steps=35)
    real_features=v30.feature_for_model
    real_geometry=v30.planned_geometry
    real_predict=v30.predict_steering
    cached={}
    count={'predictions':0,'student_model_calls':0}
    def observe_features(sim,max_gap):
        f,route=real_features(sim,max_gap)
        cached['route']=route
        return f,route
    def observe_geometry(agent,max_gap):
        geom=real_geometry(agent,max_gap)
        cached['geom']=geom
        return geom
    def predict_student(model,features):
        # Called by v30 AFTER both feature extraction and planned geometry.
        geom=cached['geom']; route=cached['route']
        pdd_carla=real_predict(model,features)
        data=state_features(features,v30.FEATURES,-float(pdd_carla),
                            geom['lateral_right_m'],geom['heading_error_rad'],route['adapted'])
        prediction=student_steer_md(student['student_model'],data,student['feature_order'],
                                    student['max_steer'],student['max_learned_residual'])
        count['predictions']+=1
        count['student_model_calls']+=1
        # V3.0 steering_action(mode=baseline) converts CARLA->MD by NEGATION.
        # Returning -steer makes the learned steering be exactly the applied MD action.
        return -prediction['steer']
    v30.feature_for_model=observe_features
    v30.planned_geometry=observe_geometry
    v30.predict_steering=predict_student
    try:
        result=v30.pilot_episode(env,seed,pdd_bundle,'baseline',opts)
    finally:
        v30.feature_for_model=real_features
        v30.planned_geometry=real_geometry
        v30.predict_steering=real_predict
    result['student_prediction_calls']=count['predictions']
    result['learned_steering_applied']=True
    result['rule_based_lateral_heading_residual_applied']=False
    return result


def evaluate(env,seed,profile,pdd_bundle,student,steps,trace_stride):
    if profile=='student_v33':
        return pilot_student_episode(env,seed,pdd_bundle,student,steps=steps,
                                     trace_stride=trace_stride)
    options=options_for(PROFILES[profile],steps=steps,trace_stride=trace_stride,tail_steps=35)
    return v30.pilot_episode(env,seed,pdd_bundle,PROFILES[profile].mode,options)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True,help='Frozen PDD V2 base model')
    p.add_argument('--student',type=Path,required=True,help='V3.3 trained student joblib')
    p.add_argument('--maps',default='C,X')
    p.add_argument('--seeds',default='28,29')
    p.add_argument('--profiles',default='baseline_3,lateral_3,student_v33')
    p.add_argument('--steps',type=int,default=1400)
    p.add_argument('--trace-stride',type=int,default=15)
    p.add_argument('--out',type=Path,default=Path('agent4_v33_closed_loop_eval.json'))
    a=p.parse_args()
    try:
        maps=[x.strip() for x in a.maps.split(',') if x.strip()]
        seeds=[int(x.strip()) for x in a.seeds.split(',') if x.strip()]
        profiles=[x.strip() for x in a.profiles.split(',') if x.strip()]
        if not maps or any(x not in ('S','C','X') for x in maps):raise ValueError('Bad maps')
        if not seeds or min(seeds)<0:raise ValueError('Bad seeds')
        if not profiles or any(x not in CHOICES for x in profiles):raise ValueError('Bad profiles')
        if any(len(set(v))!=len(v) for v in (maps,seeds,profiles)):raise ValueError('Duplicate map/seed/profile')
        if a.steps<50 or not 1<=a.trace_stride<=100:raise ValueError('Bad steps/trace stride')
    except ValueError as exc:p.error(str(exc))
    pdd=joblib.load(a.model)
    student=joblib.load(a.student)
    if list(pdd.get('features',[]))!=list(v30.FEATURES) or pdd.get('label_mode')!='at_t':
        raise ValueError('Expected PDD V2 model at_t')
    checked_student_bundle(student,v30.FEATURES,seeds)
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report={'status':'V33_CLOSED_LOOP_3MPS_A_B_NO_IDM',
            'maps':maps,'seeds':seeds,'profiles':profiles,'learner_longitudinal_used':False,
            'teacher_policy_used_in_evaluation':False,
            'student_correction':'learned model, not rule-based lateral/heading controller',
            'speed_limit_mps':3.0,'episodes_by_map':[],
            'warnings':['Only arrive_dest counts as success.',
                        'Offline teacher-action MAE does not predict closed-loop success.',
                        'All evaluations have traffic_density=0 and privileged road geometry.',
                        'Feature/geometry and slow-speed dynamics are from MetaDrive, not real vehicles.',
                        'Use eval seeds not in teacher train/validation; program checks seed overlap.']}
    for map_name in maps:
        env=None; block=[]
        try:
            env=MetaDriveEnv(dict(use_render=False,map=map_name,
                       num_scenarios=max(seeds)+1,traffic_density=0.0,
                       horizon=max(a.steps+10,1000),log_level=40))
            for seed in seeds:
                for profile in profiles:
                    result=evaluate(env,seed,profile,pdd,student,a.steps,a.trace_stride)
                    summary=episode_summary(result)
                    block.append({'seed':seed,'profile':profile,'summary':summary,
                                  'student_prediction_calls':result.get('student_prediction_calls'),
                                  'errors':result.get('errors',[]),'trace':result.get('trace',[])})
                    print(f'{map_name} seed={seed} {profile}: {summary["status"]} '
                          f'{summary["travelled_approx_m"]:.1f}m '
                          f'steps={summary["n_applied_steps"]}')
        finally:
            if env is not None:env.close()
        report['episodes_by_map'].append({'map':map_name,'episodes':block})
    report['success_by_profile']={profile:{'success':sum(1 for e in block['episodes'] if e['profile']==profile and e['summary']['status']=='arrive_dest'),
                                            'total':sum(1 for e in block['episodes'] if e['profile']==profile)}
                                   for profile in profiles for block in [dict(episodes=[e for m in report['episodes_by_map'] for e in m['episodes']])]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Saved:',a.out.resolve())
    print('Success by profile:',report['success_by_profile'])
    print('IDM policy during evaluation: NO; Original PDD longitudinal prediction: NO.')

if __name__=='__main__':main()
