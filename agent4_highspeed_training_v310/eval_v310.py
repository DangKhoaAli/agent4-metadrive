"""V3.10 held-out closed-loop A/B: PDD-only, rule teacher, learned student.

Steering of learned student = PDD + model-predicted residual. The teacher's
lateral/heading formula is NOT called for learned student evaluation. Governor
is the same across all three profiles. Only arrive_dest counts as a success.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import sys

import joblib

from common_v310 import (ROOT,VERSION,MAX_STEER,DEFAULT_SPEEDS,ALLOWED_MAPS,
                         check_pdd,check_bundle,state_features,student_action,
                         parse_seed_spec,parse_values)

sys.path.insert(0,str(ROOT/'agent4_controller_v30'))
sys.path.insert(0,str(ROOT/'agent4_controller_v31'))
import ab_closed_loop_v30 as v30
from profiles_v31 import options_for,PROFILES,episode_summary

PROFILES_V310=('pdd_only','rule_lateral','learned_v310')
STEP_S=.1


@contextmanager
def patch_student_predict(bundle,pdd_names):
    """Replace *only PDD predict method*, remain in baseline steering mode."""
    real_f=v30.feature_for_model
    real_g=v30.planned_geometry
    real_p=v30.predict_steering
    cache={}
    records=[]
    def grab_f(env,gap):
        f,route=real_f(env,gap)
        cache['features']=f;cache['route']=route
        return f,route
    def grab_g(car,gap):
        geom=real_g(car,gap)
        cache['geom']=geom
        return geom
    def predict(pdd,features):
        geom=cache.get('geom');route=cache.get('route')
        if geom is None or route is None or cache.get('features') is not features:
            raise RuntimeError('V3.10 prediction state not aligned with current observation')
        base=-float(real_p(pdd,features))
        row=state_features(features,pdd_names,base,geom['lateral_right_m'],
                           geom['heading_error_rad'],route['adapted'])
        result=student_action(bundle,row)
        records.append(result)
        return -result['steer']
    v30.feature_for_model=grab_f
    v30.planned_geometry=grab_g
    v30.predict_steering=predict
    try:yield records
    finally:
        v30.feature_for_model=real_f
        v30.planned_geometry=real_g
        v30.predict_steering=real_p



@contextmanager
def capture_governor():
    """Observe actual per-step vehicle speed, do not change the governor."""
    real=v30.governed_speed_action
    observed=[]
    def wrap(speed,target,remaining):
        action,desired=real(speed,target,remaining)
        observed.append({'speed_mps':float(speed),'desired_mps':float(desired),
                         'governor_action':float(action)})
        return action,desired
    v30.governed_speed_action=wrap
    try:yield observed
    finally:v30.governed_speed_action=real


def speed_statistics(observed,n):
    if len(observed)!=n:
        raise ValueError(f'V3.10: speed capture / action count mismatch {len(observed)} vs {n}')
    if not n:return {'max_speed_mps':None,'mean_speed_mps':None,'mean_speed_error_mps':None}
    actual=[r['speed_mps'] for r in observed]
    errors=[abs(r['speed_mps']-r['desired_mps']) for r in observed]
    return {'max_speed_mps':max(actual),'mean_speed_mps':sum(actual)/n,
            'mean_speed_error_mps':sum(errors)/n}

def run_one(env,m,seed,speed,profile,pdd,bundle,pdd_names,steps,stride):
    opts=options_for(PROFILES['lateral_3'] if profile=='rule_lateral' else
                     PROFILES['baseline_3'],steps=steps,trace_stride=stride,tail_steps=35)
    opts.speed_limit_mps=float(speed)
    with capture_governor() as governor_steps:
        if profile=='learned_v310':
            with patch_student_predict(bundle,pdd_names) as records:
                result=v30.pilot_episode(env,seed,pdd,'baseline',opts)
        else:
            records=[]
            result=v30.pilot_episode(env,seed,pdd,'residual' if profile=='rule_lateral' else 'baseline',opts)
    summary=episode_summary(result)
    n=result['n_applied_steps']
    try:summary.update(speed_statistics(governor_steps,n))
    except ValueError as exc:
        result.setdefault('errors',[]).append(str(exc))
        summary['error_count']=len(result['errors'])
    if profile=='learned_v310':
        summary['student_prediction_calls']=len(records)
        summary['student_predictions_aligned']=len(records)==n
        summary['model_residual_nonzero_steps']=sum(abs(z['learned_residual'])>1e-8 for z in records)
        summary['max_abs_model_residual']=max((abs(z['learned_residual']) for z in records),default=0.)
        if len(records)!=n:
            result.setdefault('errors',[]).append(f'Prediction calls {len(records)} != applied actions {n}')
            summary['error_count']=len(result['errors'])
        for sample in result.get('trace',[]):
            i=sample['step']
            if i>=len(records):continue
            r=records[i]
            sample['v310_attribution']={'pdd_base':r['base'],
               'model_residual':r['learned_residual'],
               'model_proposed_action':r['steer'],
               'actual_applied_action':sample['action_applied'][0]}
    summary.update(sim_elapsed_s=n*STEP_S,
        time_to_goal_s=n*STEP_S if result['status']=='arrive_dest' else None,
        requested_speed_mps=float(speed),
        learned_steering_applied=profile=='learned_v310',
        teacher_fallback_calls=0,
        learned_longitudinal_applied=False)
    return {'map':m,'seed':int(seed),'speed_mps':float(speed),'profile':profile,
            'summary':summary,'errors':result.get('errors',[]),
            'trace':result.get('trace',[])}


def scores(episodes):
    groups=defaultdict(list)
    for e in episodes:groups[(e['map'],e['speed_mps'],e['profile'])].append(e)
    return [dict(map=m,speed_mps=s,profile=p,
          success=sum(x['summary']['status']=='arrive_dest' for x in grp),
          total=len(grp),
          mean_success_time_s=(sum(x['summary']['time_to_goal_s'] for x in grp if x['summary']['time_to_goal_s'] is not None)/sum(x['summary']['time_to_goal_s'] is not None for x in grp)
                   if any(x['summary']['time_to_goal_s'] is not None for x in grp) else None))
          for (m,s,p),grp in sorted(groups.items())]


def run(args):
    maps=parse_values(args.maps,ALLOWED_MAPS)
    speeds=parse_values(args.speeds,DEFAULT_SPEEDS,float)
    seeds=parse_seed_spec(args.seeds)
    profiles=parse_values(args.profiles,PROFILES_V310)
    if not 100<=args.steps<=5000 or not 1<=args.trace_stride<=100:
        raise ValueError('Steps 100..5000, stride 1..100')
    if min(seeds)<146:raise ValueError('V3.10 holdout seeds must be >=146')
    pdd=joblib.load(args.pdd)
    pdd_names=check_pdd(pdd)
    bundle=joblib.load(args.steering)
    check_bundle(bundle,pdd_names,seeds)
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report={'version':VERSION,'maps':maps,'speeds_mps':speeds,'seeds':seeds,
      'profiles':profiles,'traffic_density':0.,'frozen_pdd':True,
      'model_training_performed':False,'learned_longitudinal_used':False,
      'warnings':['No model-governor longitudinal in this test; do not claim end-to-end learned autonomy.',
         'PDD and rule teacher are independent baselines; learner has NO teacher steering fallback.',
         'Only zero-traffic, privileged-road-geometry MetaDrive simulations.',
         'No evaluation on C/X curves or traffic. PDD baseline may be better in closed loop.'],
      'episodes':[]}
    for m in maps:
        env=MetaDriveEnv(dict(use_render=False,map=m,num_scenarios=max(seeds)+1,
            traffic_density=0.,horizon=max(1000,args.steps+10),log_level=40,
            physics_world_step_size=.02,decision_repeat=5))
        try:
            for seed in seeds:
                for speed in speeds:
                    for profile in profiles:
                        e=run_one(env,m,seed,speed,profile,pdd,bundle,pdd_names,
                                  args.steps,args.trace_stride)
                        report['episodes'].append(e)
                        sm=e['summary'];print(f'{m} seed={seed} speed={speed:g} '
                            f'{profile:13s} {sm["status"]:30s} lateral={sm["max_abs_planned_lateral_right_m"]:.3f} '
                            f'time={sm["time_to_goal_s"]}',flush=True)
        finally:env.close()
    report['scores']=scores(report['episodes'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False),encoding='utf-8')
    outcsv=args.out.with_suffix('.csv')
    cols=['map','seed','speed_mps','profile','status','success','time_to_goal_s',
          'max_speed_mps','mean_speed_mps',
          'max_abs_planned_lateral_right_m','min_estimated_edge_clearance_m',
          'near_edge_warning_steps','error_count','student_predictions_aligned']
    with outcsv.open('w',newline='',encoding='utf-8-sig') as fp:
        writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
        for e in report['episodes']:
            s=e['summary']; row={k:e[k] for k in ('map','seed','speed_mps','profile')}
            row.update(status=s['status'],success=int(s['status']=='arrive_dest'))
            for k in cols:
                if k not in row:row[k]=s.get(k)
            writer.writerow(row)
    print('JSON:',args.out.resolve(),'CSV:',outcsv.resolve())
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',required=True,type=Path)
    p.add_argument('--steering',required=True,type=Path)
    p.add_argument('--maps',default='S,SSSSS')
    p.add_argument('--seeds',default='146,147')
    p.add_argument('--speeds',default='4,6,8,10')
    p.add_argument('--profiles',default='pdd_only,rule_lateral,learned_v310')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--trace-stride',type=int,default=30)
    p.add_argument('--out',type=Path,default=Path('agent4_v310_closed_loop_eval.json'))
    return run(p.parse_args(argv))

if __name__=='__main__':main()
