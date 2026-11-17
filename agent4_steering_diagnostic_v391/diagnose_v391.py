"""Agent 4 V3.9.1 - diagnose LONG-ROAD lateral drift with unchanged governor.

PDD_ONLY, RULE_LATERAL, LEARNED_V35 all run on same MetaDrive maps/seeds/speeds.
No model retraining, no traffic, no learned longitudinal. Simulation only.
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

ROOT=Path(__file__).resolve().parent.parent
for folder,filename in [
    ('agent4_robustness_v34','benchmark_v34.py'),
    ('agent4_multispeed_v35','evaluate_multispeed_v35.py'),
]:
    item=ROOT/folder/filename
    if not item.is_file():
        raise RuntimeError(f'Missing project dependency: {item}')
    sys.path.insert(0,str(item.parent))

import benchmark_v34 as v34
from evaluate_multispeed_v35 import check_student

VERSION='V391_FROZEN_STEERING_CAUSAL_ABLATION'
PROFILES={'pdd_only':'baseline_3',
          'rule_lateral':'lateral_3',
          'learned_v35':'student_v33'}
MAPS=('S','SS','SSS','SSSS','SSSSS','C','X')
STEP_S=0.1


def speeds_csv(spec):
    try: speeds=[float(x.strip()) for x in spec.split(',') if x.strip()]
    except ValueError as e: raise ValueError('Invalid speeds list') from e
    if not speeds or len(set(speeds))!=len(speeds) or not all(
            math.isfinite(v) and .5<=v<=12 for v in speeds):
        raise ValueError('Speeds must be distinct finite numbers 0.5..12 m/s')
    return speeds


@contextmanager
def observe_governor():
    """Observe speed without changing governor output or end-to-end action."""
    old=v34.v30.governed_speed_action
    samples=[]
    def wrapper(speed,target,remaining):
        action,desired=old(speed,target,remaining)
        samples.append({'speed_mps':float(speed),'target_mps':float(desired),
                        'governor_action':float(action)})
        return action,desired
    v34.v30.governed_speed_action=wrapper
    try:yield samples
    finally:v34.v30.governed_speed_action=old


def speeds_stats(samples,n):
    if n!=len(samples):
        raise RuntimeError(f'Governor observation mismatch: {len(samples)} vs {n} actions')
    if not samples:
        return dict(max_speed_mps=None,mean_speed_mps=None,mean_abs_speed_error_mps=None)
    speeds=[x['speed_mps'] for x in samples]
    targets=[x['target_mps'] for x in samples]
    if not all(math.isfinite(x) for x in speeds+targets):
        raise ValueError('Non-finite speeds in governor capture')
    return dict(max_speed_mps=max(speeds),mean_speed_mps=sum(speeds)/n,
                mean_abs_speed_error_mps=sum(abs(a-b) for a,b in zip(speeds,targets))/n)


def run_one(env,map_name,seed,speed,profile,pdd,student,steps,stride):
    with observe_governor() as samples:
        result=v34.eval_one(env,map_name,seed,speed,PROFILES[profile],pdd,
                            student,steps,stride)
    s=result['summary']
    s.update(speeds_stats(samples,s['n_applied_steps']))
    s['sim_elapsed_s']=s['n_applied_steps']*STEP_S
    s['time_to_goal_s']=(s['sim_elapsed_s'] if s['status']=='arrive_dest' else None)
    s['requested_speed_mps']=speed
    s['outside_v35_training_speed']=bool(speed>4. or speed<2.)
    s['learned_steering_applied']=profile=='learned_v35'
    s['rule_lateral_heading_residual_applied']=profile=='rule_lateral'
    s['learned_longitudinal_applied']=False
    trace=result.get('trace',[])
    s['initial_lane_index']=trace[0].get('lane_index') if trace else None
    s['terminal_lane_index']=trace[-1].get('lane_index') if trace else None
    result['profile']=profile
    return result


def score(episodes):
    grouped=defaultdict(list)
    for e in episodes:
        grouped[(e['map'],e['speed_mps'],e['profile'])].append(e)
    return [dict(map=m,speed_mps=v,profile=p,
                 success=sum(e['summary']['status']=='arrive_dest' for e in subset),
                 total=len(subset),
                 failures=[dict(seed=e['seed'],status=e['summary']['status'])
                           for e in subset if e['summary']['status']!='arrive_dest'])
            for (m,v,p),subset in sorted(grouped.items())]


def write_csv(report,path):
    header=['map','seed','speed_mps','profile','status','success','time_to_goal_s',
            'sim_elapsed_s','max_speed_mps','mean_speed_mps',
            'max_abs_planned_lateral_right_m','min_estimated_edge_clearance_m',
            'steering_slew_limited_steps','initial_lane_index','terminal_lane_index',
            'error_count']
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=header);writer.writeheader()
        for e in report['episodes']:
            s=e['summary']
            row={'map':e['map'],'seed':e['seed'],'speed_mps':e['speed_mps'],
                 'profile':e['profile'],'status':s['status'],
                 'success':int(s['status']=='arrive_dest')}
            for k in header:
                if k not in row:row[k]=s.get(k)
            writer.writerow(row)


def run(args):
    maps=v34.parse_simple_csv(args.maps,MAPS)
    seeds=v34.parse_seeds(args.seeds)
    speeds=speeds_csv(args.speeds)
    profiles=v34.parse_simple_csv(args.profiles,tuple(PROFILES))
    if min(seeds)<122:
        raise ValueError('Reserve diagnostic seeds >=122, separate from V3.9 seeds 120/121')
    if not (100<=args.steps<=5000 and 1<=args.trace_stride<=100):
        raise ValueError('steps must be 100..5000; trace-stride must be 1..100')
    pdd=joblib.load(args.pdd)
    student=joblib.load(args.steering)
    if pdd.get('label_mode')!='at_t' or list(pdd.get('features',[]))!=list(v34.v30.FEATURES):
        raise ValueError('Expected PDD at_t with exact 15 features')
    check_student(student,pdd['features'],seeds,'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report=dict(version=VERSION,maps=maps,seeds=seeds,speeds_mps=speeds,
                profiles=profiles,traffic_density=0.0,
                frozen_pdd=True,frozen_v35_steering=True,
                governor_unchanged=True,model_training_performed=False,
                warnings=[
                    'Changing steering profile isolates controller effects, not a proof of causal independence under closed-loop feedback.',
                    'Guard lane_departure_guard is a diagnostic threshold; out_of_road is actual simulator termination.',
                    'Steering V3.5 is trained only for speeds 2..4 m/s; higher speeds are OOD.',
                    'No traffic, using privileged navigation geometry, not physical-car evidence.'
                ],episodes=[])
    for m in maps:
        env=MetaDriveEnv(dict(use_render=False,map=m,num_scenarios=max(seeds)+1,
                              traffic_density=0.0,horizon=max(1000,args.steps+10),
                              log_level=40,physics_world_step_size=0.02,decision_repeat=5))
        try:
            for seed in seeds:
                for speed in speeds:
                    for p in profiles:
                        e=run_one(env,m,seed,speed,p,pdd,student,args.steps,args.trace_stride)
                        report['episodes'].append(e)
                        s=e['summary']
                        vmax=s['max_speed_mps']
                        print(f'{m:5} seed={seed} target={speed:g} {p:13s} '
                              f'{s["status"]:27s} distance={s["travelled_approx_m"]:.1f}m '
                              f'vmax={vmax if vmax is not None else "NA"}',flush=True)
        finally:env.close()
    report['group_scores']=score(report['episodes'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(report,args.out.with_suffix('.csv'))
    print('JSON:',args.out.resolve())
    print('CSV:',args.out.with_suffix('.csv').resolve())
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',type=Path,required=True)
    p.add_argument('--steering',type=Path,required=True)
    p.add_argument('--maps',default='SSSSS',help='S,SS,SSS,SSSS,SSSSS,C,X')
    p.add_argument('--seeds',default='122,123')
    p.add_argument('--speeds',default='4,6',help='2,3,4,6,8,10; >4 is untrained')
    p.add_argument('--profiles',default='pdd_only,rule_lateral,learned_v35')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--trace-stride',type=int,default=20)
    p.add_argument('--out',type=Path,default=Path('agent4_v391_steering_ablation.json'))
    return run(p.parse_args(argv))


if __name__=='__main__':main()
