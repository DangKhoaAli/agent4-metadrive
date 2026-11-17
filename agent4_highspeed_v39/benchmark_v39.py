"""Agent4 V3.9: frozen-policy high-speed, EMPTY-road diagnostic (not training).

Three equal-map/seed policies:
  governor_v35 = frozen learned V3.5 steering + unchanged rule speed governor
  learned_v37 = frozen V3.5 steering + frozen V3.7 learned signed throttle/brake
  learned_v38 = frozen V3.5 steering + frozen V3.8 learned traffic longitudinal
All >4 m/s tests are extrapolation, not evidence of previously trained competence.
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
for dependency in ('agent4_robustness_v34','agent4_multispeed_v35',
                   'agent4_onpolicy_longitudinal_v37','agent4_traffic_v38'):
    path=ROOT/dependency
    if not path.is_dir():
        raise RuntimeError(f'Missing sibling project directory: {path}')
    sys.path.insert(0,str(path))

import benchmark_v34 as v34
from evaluate_multispeed_v35 import check_student
from common_v37 import VERSION as V37_VERSION,names as v37_names
from common_v38 import VERSION as V38_VERSION,feature_names as v38_names
from traffic_v38 import captured_longitudinal

PROFILES=('governor_v35','learned_v37','learned_v38')
# MetaDrive's PG map config accepts a string of block codes. Larger S strings
# are useful to observe sustained speed before the goal slowdown begins.
MAPS=('S','SS','SSS','SSSS','SSSSS','C','X')
SIM_STEP_S=0.02*5  # explicit physics step and decision repeat in env config


def parse_speeds_v39(value: str) -> list[float]:
    """A deliberately NEW parser: V3.4's CLI helper caps input at 5 m/s."""
    try:
        result=[float(x.strip()) for x in value.split(',') if x.strip()]
    except ValueError as exc:
        raise ValueError('Speeds must be comma-separated finite numbers') from exc
    if not result or len(result)!=len(set(result)) or not all(
            math.isfinite(x) and .5<=x<=12. for x in result):
        raise ValueError('Choose non-repeating finite speeds between 0.5 and 12 m/s')
    return result


def validate_bundles(pdd,steering,v37,v38,seeds):
    if pdd.get('label_mode')!='at_t' or list(pdd.get('features',[]))!=list(v34.v30.FEATURES):
        raise ValueError('Expected PDD V2 at_t, 15 features in exact order')
    check_student(steering,pdd['features'],seeds,'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    if v37.get('version')!=V37_VERSION or v37.get('feature_order')!=v37_names(pdd['features']):
        raise ValueError('V3.7 model schema mismatch')
    if v38.get('version')!=V38_VERSION or v38.get('feature_order')!=v38_names(pdd['features']):
        raise ValueError('V3.8 model schema mismatch')
    for key,bundle in [('V3.7',v37),('V3.8',v38)]:
        groups=bundle.get('splits',{})
        trainval={int(s) for g in ('train','val') for s in groups.get(g,[])}
        conflict=set(seeds)&trainval
        if conflict:raise ValueError(f'{key}: eval seed leaked into train/val: {sorted(conflict)}')
        if not hasattr(bundle.get('model'),'predict'):
            raise ValueError(f'{key}: missing estimator')
    if min(seeds)<120:
        raise ValueError('V3.9 exploration reserves seeds >=120; previous V3.8 eval ends at 119')


@contextmanager
def capture_governor(records):
    """Record the EXACT governor applied at every episode step, no substitution."""
    old=v34.v30.governed_speed_action
    def callback(speed,target,remaining):
        applied,desired=old(speed,target,remaining)
        records.append(dict(step=len(records),speed_mps_before=float(speed),
                            route_target_mps=float(desired),
                            baseline_governor_action=float(applied),
                            applied_longitudinal=float(applied),
                            front_present=False))
        return applied,desired
    v34.v30.governed_speed_action=callback
    try:yield
    finally:v34.v30.governed_speed_action=old


def metrics_from_actions(actions,n,sim_step_s=SIM_STEP_S):
    if len(actions)!=n:
        raise RuntimeError(f'Longitudinal capture mismatch: {len(actions)} calls / {n} applied steps')
    if not actions:
        return dict(mean_speed_mps=None,max_speed_mps=None,sim_elapsed_s=None,
                    brake_steps=0,throttle_steps=0,governor_deviation_mae=None)
    speeds=[float(a['speed_mps_before']) for a in actions]
    desired=[float(a['route_target_mps']) for a in actions]
    applied=[float(a['applied_longitudinal']) for a in actions]
    ref=[float(a['baseline_governor_action']) for a in actions]
    if not all(math.isfinite(x) for x in speeds+desired+applied+ref):
        raise RuntimeError('Nonfinite control/speed in high-speed benchmark')
    if any(a.get('front_present',False) for a in actions):
        raise RuntimeError('No-traffic benchmark invalid: observed another vehicle')
    if any(abs(x)>1.000001 for x in applied):
        raise RuntimeError('Applied action outside normalized MetaDrive range')
    return dict(mean_speed_mps=sum(speeds)/n,max_speed_mps=max(speeds),
                sim_elapsed_s=round(n*sim_step_s,6),
                mean_abs_speed_tracking_error_mps=sum(abs(s-t) for s,t in zip(speeds,desired))/n,
                max_overspeed_mps=max(max(0,s-t) for s,t in zip(speeds,desired)),
                target_speed_reached_steps=sum(s>=t-.5 and t>=.9*max(desired) for s,t in zip(speeds,desired)),
                brake_steps=sum(x<-.02 for x in applied),
                throttle_steps=sum(x>.02 for x in applied),
                governor_deviation_mae=sum(abs(x-y) for x,y in zip(applied,ref))/n)


def run_episode(env,map_name,seed,speed,profile,pdd,steering,old,new,steps,stride):
    actions=[]
    if profile=='governor_v35':
        context=capture_governor(actions)
    elif profile=='learned_v37' or profile=='learned_v38':
        # Reuse exact, previously smoke-tested V3.8 predictor hook. No lead is
        # spawned, no teacher fallback, and both policies see front_present=0.
        raw=[]
        context=captured_longitudinal(env,pdd['features'],profile,old,new,raw)
    else:
        raise ValueError(f'Unknown profile {profile}')
    with context:
        result=v34.eval_one(env,map_name,seed,speed,'student_v33',pdd,steering,steps,stride)
    if profile!='governor_v35':
        for i,r in enumerate(raw):
            actions.append(dict(step=i,speed_mps_before=float(r['row']['speed_mps']),
                                route_target_mps=float(r['route_target_mps']),
                                baseline_governor_action=float(r['baseline_governor_action']),
                                applied_longitudinal=float(r['applied_longitudinal']),
                                front_present=bool(r['front']['present'])))
    n=result['summary']['n_applied_steps']
    extended=metrics_from_actions(actions,n)
    # Trace/action pairing makes sure reported forecasts actually drove the car.
    for event in result['trace']:
        i=event['step']
        if i>=len(actions) or abs(float(event['action_applied'][1])-actions[i]['applied_longitudinal'])>1e-8:
            raise RuntimeError(f'Longitudinal trace/action misalignment at step {i}')
        event['longitudinal_v39']={k:actions[i][k] for k in (
            'speed_mps_before','route_target_mps','applied_longitudinal','baseline_governor_action')}
    result['profile']=profile
    result['summary'].update(extended)
    result['summary'].update(
        frozen_steering_v35=True,
        learned_longitudinal_applied=profile!='governor_v35',
        longitudinal_prediction_calls=n if profile!='governor_v35' else 0,
        longitudinal_actions_aligned=True,
        outside_learned_training_speeds=bool(speed>4. or speed<2.),
        requested_cruise_speed_mps=float(speed),
        requested_cruise_speed_kmh=round(speed*3.6,4),
        time_to_goal_s=extended['sim_elapsed_s'] if result['summary']['status']=='arrive_dest' else None)
    return result


def group_statistics(episodes):
    groups=defaultdict(list)
    for e in episodes:
        groups[(e['map'],e['speed_mps'],e['profile'])].append(e)
    ans=[]
    for (map_name,speed,profile),subset in sorted(groups.items()):
        arrived=[e for e in subset if e['summary']['status']=='arrive_dest']
        times=[e['summary']['time_to_goal_s'] for e in arrived]
        ans.append(dict(map=map_name,speed_mps=speed,profile=profile,
                        success=len(arrived),total=len(subset),
                        success_rate=len(arrived)/len(subset),
                        mean_time_to_goal_success_only_s=(sum(times)/len(times) if times else None),
                        min_time_to_goal_success_only_s=min(times,default=None),
                        max_observed_speed_mps=max((e['summary']['max_speed_mps'] or 0) for e in subset),
                        failure_statuses=sorted({e['summary']['status'] for e in subset if
                                                e['summary']['status']!='arrive_dest'})))
    return ans


def write_csv(report,out):
    headers=['map','seed','speed_mps','profile','status','success','time_to_goal_s',
             'sim_elapsed_s','n_applied_steps','max_speed_mps','mean_speed_mps',
             'max_abs_planned_lateral_right_m','min_estimated_edge_clearance_m',
             'near_edge_warning_steps','brake_steps','throttle_steps','terminal_reasons',
             'outside_learned_training_speeds']
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader()
        for e in report['episodes']:
            s=e['summary']
            row=dict(map=e['map'],seed=e['seed'],speed_mps=e['speed_mps'],profile=e['profile'],
                status=s['status'],success=int(s['status']=='arrive_dest'),
                terminal_reasons=';'.join(s['terminal_reasons']))
            row.update({k:s.get(k) for k in headers if k not in row})
            writer.writerow(row)


def benchmark(a):
    maps=v34.parse_simple_csv(a.maps,MAPS)
    seeds=v34.parse_seeds(a.seeds)
    speeds=parse_speeds_v39(a.speeds)
    profiles=v34.parse_simple_csv(a.profiles,PROFILES)
    if not 100<=a.steps<=5000 or not 1<=a.trace_stride<=100:
        raise ValueError('Steps must be 100..5000 and trace-stride 1..100')
    pdd=joblib.load(a.pdd);steering=joblib.load(a.steering)
    v37=joblib.load(a.longitudinal_v37);v38=joblib.load(a.longitudinal_v38)
    validate_bundles(pdd,steering,v37,v38,seeds)
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report=dict(version='V39_HIGH_SPEED_UNTRAINED_FROZEN_DIAGNOSTIC',
                traffic_density=0.,no_scripted_lead=True,nav_geometry_privileged=True,
                dynamics_dt_s=SIM_STEP_S,physics_world_step_size=0.02,decision_repeat=5,
                maps=maps,seeds=seeds,speeds_mps=speeds,profiles=profiles,
                frozen_pdd=True,frozen_steering_v35=True,
                frozen_longitudinal_v37_v38=True,training_performed=False,
                warnings=['4 m/s is the maximum speed with prior learned model training.',
                          '6/8/10 m/s and any other speed beyond [2,4] are OOD experiments.',
                          'V3.5 steering is also untrained above 4 m/s.',
                          'No traffic and privileged simulator features only.',
                          'Only arrive_dest is a success; faster failure never outranks arrival.',
                          'Use newly reserved seeds for future training and independent validation.',
                          'Not evidence of safety for real vehicles.'],episodes=[])
    for map_name in maps:
        env=MetaDriveEnv(dict(use_render=False,map=map_name,num_scenarios=max(seeds)+1,
               traffic_density=0.,horizon=max(1000,a.steps+10),log_level=40,
               physics_world_step_size=0.02,decision_repeat=5))
        try:
            for seed in seeds:
                for speed in speeds:
                    for profile in profiles:
                        e=run_episode(env,map_name,seed,speed,profile,pdd,steering,v37,v38,
                                      a.steps,a.trace_stride)
                        report['episodes'].append(e)
                        s=e['summary']
                        print(f'{map_name:5s} seed={seed} vtarget={speed:4.1f} '
                              f'{profile:13s} {s["status"]:28s} '
                              f'vmax={s["max_speed_mps"] or 0:5.2f} m/s '
                              f'tgoal={s["time_to_goal_s"]}s',flush=True)
        finally:
            env.close()
    report['group_scores']=group_statistics(report['episodes'])
    report['overall_by_profile']={p:{'success':sum(e['profile']==p and e['summary']['status']=='arrive_dest' for e in report['episodes']),
                                     'total':sum(e['profile']==p for e in report['episodes'])} for p in profiles}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    csv_out=a.csv_out or a.out.with_suffix('.csv')
    write_csv(report,csv_out)
    print('Saved JSON:',a.out.resolve())
    print('Saved CSV:',csv_out.resolve())
    print('Scores:',report['overall_by_profile'])
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',required=True,type=Path)
    p.add_argument('--steering',required=True,type=Path)
    p.add_argument('--longitudinal-v37',required=True,type=Path)
    p.add_argument('--longitudinal-v38',required=True,type=Path)
    p.add_argument('--maps',default='S,C,X',help='S,SS,SSS,SSSS,SSSSS,C,X')
    p.add_argument('--seeds',default='120,121',help='Fresh diagnostic seeds >=120')
    p.add_argument('--speeds',default='4,6,8,10',help='Desired speed in m/s, max 12')
    p.add_argument('--profiles',default=','.join(PROFILES))
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--trace-stride',type=int,default=30)
    p.add_argument('--out',type=Path,default=Path('agent4_v39_speed_sweep.json'))
    p.add_argument('--csv-out',type=Path,default=None)
    a=p.parse_args(argv)
    try:benchmark(a)
    except ValueError as exc:p.error(str(exc))

if __name__=='__main__':main()
