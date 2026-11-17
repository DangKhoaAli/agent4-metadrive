"""V3.6 closed-loop A/B: SAME frozen V3.5 steering and SAME desired speed.

A: V3.5 steering + original rule governor longitudinal.
B: V3.5 steering + V3.6 predicted longitudinal (-1 brake, +1 throttle).
No IDMPolicy during evaluation and no governor-action fallback in mode B.
Uses V3.0 physical-road and departure guards unchanged; simulator only.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from contextlib import contextmanager
import joblib

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_multispeed_v35'))
from evaluate_multispeed_v35 import check_student
sys.path.insert(0,str(ROOT/'agent4_robustness_v34'))
from benchmark_v34 import (parse_seeds,parse_speeds,parse_simple_csv,eval_one,
                            summary_rows,group_scores,v30)
from common_v36 import (MAPS,VERSION,make_row,names,predict_action,check_eval_seeds)

PROFILES=('governor_v35','learned_long_v36')


@contextmanager
def longitudinal_control(mode,student_bundle,pdd_features,actions):
    """Patch only low-level longitudinal command per episode; always restore."""
    if mode not in PROFILES:raise ValueError(mode)
    real_f=v30.feature_for_model
    real_g=v30.planned_geometry
    real_long=v30.governed_speed_action
    cached={}
    def observe(env,max_gap):
        features,route=real_f(env,max_gap)
        cached['features']=features
        cached['route']=route
        return features,route
    def observe_geom(car,max_gap):
        geom=real_g(car,max_gap)
        cached['geom']=geom
        return geom
    def apply(speed,base_speed,remaining):
        baseline_action,target=real_long(speed,base_speed,remaining)
        if mode=='governor_v35':
            applied=float(baseline_action)
        else:
            row=make_row(cached['features'],pdd_features,cached['geom'],
                 cached['route']['adapted'],target,cached['route']['remaining_m'])
            applied=predict_action(student_bundle,row)
        actions.append({'governor_action':float(baseline_action),
                        'applied_longitudinal':applied,'target_speed_mps':float(target),
                        'speed_mps_before':float(speed),
                        'remaining_route_m':float(remaining)})
        return applied,float(target)
    v30.feature_for_model=observe
    v30.planned_geometry=observe_geom
    v30.governed_speed_action=apply
    try:yield
    finally:
        v30.feature_for_model=real_f
        v30.planned_geometry=real_g
        v30.governed_speed_action=real_long


def one(env,map_name,seed,speed,profile,pdd,steering,long_bundle,steps,stride):
    actions=[]
    with longitudinal_control(profile,long_bundle,pdd['features'],actions):
        item=eval_one(env,map_name,seed,speed,'student_v33',pdd,steering,steps,stride)
    item['profile']=profile
    n=item['summary']['n_applied_steps']
    item['summary']['longitudinal_prediction_calls']=len(actions) if profile=='learned_long_v36' else 0
    item['summary']['longitudinal_actions_aligned']=(len(actions)==n)
    item['summary']['learned_longitudinal_applied']=profile=='learned_long_v36'
    item['summary']['brake_applied_steps']=sum(a['applied_longitudinal']<-.02 for a in actions[:n])
    item['summary']['throttle_applied_steps']=sum(a['applied_longitudinal']>.02 for a in actions[:n])
    item['summary']['mean_abs_speed_tracking_error_mps']=(
        sum(abs(a['speed_mps_before']-a['target_speed_mps']) for a in actions[:n])/n if n else None)
    item['summary']['max_observed_overspeed_mps']=max([max(0,a['speed_mps_before']-a['target_speed_mps']) for a in actions[:n]],default=0)
    item['summary']['max_observed_speed_mps']=max([a['speed_mps_before'] for a in actions[:n]],default=0)
    # V3.4 trace stores time-strided rows. Attach the true longitudinal source.
    for event in item['trace']:
        idx=event.get('step')
        if isinstance(idx,int) and idx<len(actions):
            a=actions[idx]
            event['longitudinal_attribution_v36']={
                'mode':profile,'governor_reference_action':a['governor_action'],
                'learned_action_applied':a['applied_longitudinal'] if profile=='learned_long_v36' else None,
                'actual_longitudinal_action':float(event['action_applied'][1]),
                'desired_speed_mps':a['target_speed_mps']}
    if len(actions)!=n:
        item['errors'].append(f'Longitudinal command/application mismatch: {len(actions)} vs {n}')
        item['summary']['error_count']=len(item['errors'])
    return item


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',required=True,type=Path)
    p.add_argument('--steering',required=True,type=Path,help='Frozen V3.5 steering student')
    p.add_argument('--longitudinal',required=True,type=Path,help='Trained V3.6 model')
    p.add_argument('--maps',default='C,X')
    p.add_argument('--seeds',default='76,77')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--profiles',default='governor_v35,learned_long_v36')
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--trace-stride',type=int,default=20)
    p.add_argument('--out',type=Path,default=Path('agent4_v36_closed_loop_eval.json'))
    a=p.parse_args(argv)
    try:
        maps=parse_simple_csv(a.maps,MAPS)
        seeds=parse_seeds(a.seeds)
        speeds=parse_speeds(a.speeds)
        profiles=parse_simple_csv(a.profiles,PROFILES)
        if not 100<=a.steps<=5000 or not 1<=a.trace_stride<=100:p.error('Bad steps/trace stride')
    except ValueError as exc:p.error(str(exc))
    pdd=joblib.load(a.pdd)
    steering=joblib.load(a.steering)
    long=joblib.load(a.longitudinal)
    if pdd.get('label_mode')!='at_t' or len(pdd.get('features',[]))!=15:
        raise ValueError('Expected frozen PDD V2 at_t (15 features)')
    check_student(steering,pdd['features'],seeds,'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    if long.get('version')!=VERSION or long.get('feature_order')!=names(pdd['features']):
        raise ValueError('Longitudinal model version/feature mismatch')
    if long.get('pdd_features')!=pdd['features']:raise ValueError('Longitudinal PDD contract mismatch')
    check_eval_seeds(long,seeds)
    if not set(speeds)<=set(long['speed_targets']):
        raise ValueError('Requested evaluation speed outside V3.6 training speeds')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report={'status':'V36_CLOSED_LOOP_STEERING_V35_LONGITUDINAL_AB',
            'maps':maps,'seeds':seeds,'speeds_mps':speeds,'profiles':profiles,
            'idm_used_for_control':False,'traffic_density':0.0,
            'same_v35_steering_in_both_profiles':True,
            'governor_action_used_by_learned_profile':False,
            'longitudinal_model_active_in_learned_profile':True,
            'warnings':['Only arrive_dest counts as success.',
                        'Privileged simulator state and zero traffic; not a physical safety evaluation.',
                        'Target speed and final-route slowdown still use a hand-designed planner.',
                        'Student steering V3.5 frozen, PDD base frozen.',
                        'Offline loss does not imply closed-loop success.'],
            'episodes':[]}
    for m in maps:
        env=None
        try:
            env=MetaDriveEnv(dict(use_render=False,map=m,
                   num_scenarios=max(seeds)+1,traffic_density=0.0,
                   horizon=max(1000,a.steps+10),log_level=40))
            for seed in seeds:
                for speed in speeds:
                    for profile in profiles:
                        item=one(env,m,seed,speed,profile,pdd,steering,long,a.steps,a.trace_stride)
                        report['episodes'].append(item)
                        s=item['summary']
                        print(f'{m} seed={seed} speed={speed:g} {profile}: '
                              f'{s["status"]} {s["travelled_approx_m"]:.1f}m '
                              f'overspeed_max={s["max_observed_overspeed_mps"]:.2f}',flush=True)
        finally:
            if env is not None:env.close()
    report['success_by_profile']={p:{'success':sum(e['profile']==p and e['summary']['status']=='arrive_dest' for e in report['episodes']),
                                     'total':sum(e['profile']==p for e in report['episodes'])} for p in profiles}
    rows=summary_rows(report['episodes'])
    report['scores_by_speed_map_profile']=group_scores(rows)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Output:',a.out.resolve())
    print('Success:',report['success_by_profile'])

if __name__=='__main__':main()
