"""Same steering/seed/lead schedule: scripted teacher vs V3.7 vs learned V3.8."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_robustness_v34'))
from benchmark_v34 import parse_seeds,parse_speeds,parse_simple_csv,v30
sys.path.insert(0,str(ROOT/'agent4_multispeed_v35'))
from evaluate_multispeed_v35 import check_student
sys.path.insert(0,str(ROOT/'agent4_onpolicy_longitudinal_v37'))
from common_v37 import VERSION as OLD_VERSION,names as old_names
from common_v38 import VERSION,SCENARIOS,feature_names
from traffic_v38 import run_one
PROFILES=('teacher_v38','learned_v37','learned_v38')


def evaluate(a):
    maps=parse_simple_csv(a.maps,('S',));seeds=parse_seeds(a.seeds)
    speeds=parse_speeds(a.speeds);scenarios=parse_simple_csv(a.scenarios,SCENARIOS)
    profiles=parse_simple_csv(a.profiles,PROFILES)
    if min(seeds)<100:raise ValueError('New evaluation seeds >=100 required')
    pdd=joblib.load(a.pdd);steering=joblib.load(a.steering)
    old=joblib.load(a.old_longitudinal);new=joblib.load(a.longitudinal)
    if pdd.get('features')!=list(v30.FEATURES) or pdd.get('label_mode')!='at_t':
        raise ValueError('Wrong PDD features')
    check_student(steering,pdd['features'],seeds,'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    if old.get('version')!=OLD_VERSION or old.get('feature_order')!=old_names(pdd['features']):
        raise ValueError('Bad V3.7 reference model')
    if new.get('version')!=VERSION or new.get('feature_order')!=feature_names(pdd['features']):
        raise ValueError('Bad V3.8 traffic model')
    if set(seeds)&(set(new['splits']['train'])|set(new['splits']['val'])|
                   set(old['splits']['train'])|set(old['splits']['val'])):
        raise ValueError('Evaluation seed leaked to training or validation')
    if not set(speeds)<=set(new['speeds_mps']) or not set(scenarios)<=set(new['scenarios']):
        raise ValueError('Scenario/speed outside training conditions')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report=dict(status='V38_TRAFFIC_CLOSED_LOOP_AB_C',maps=maps,seeds=seeds,
                speeds_mps=speeds,scenarios=scenarios,profiles=profiles,
                controlled_lead=True,scripted_teacher_not_idm=True,
                learned_v38_no_teacher_fallback=True,episodes=[],
                warnings=['Only arrive_dest counts as route success.',
                          'Traffic is one controlled lead, not random traffic.',
                          'All observation features are privileged simulator data.',
                          'No collision-free guarantee or public-road applicability.'])
    for map_name in maps:
        env=MetaDriveEnv(dict(use_render=False,map=map_name,
                   num_scenarios=max(seeds)+1,traffic_density=0.,
                   horizon=max(1000,a.steps+10),log_level=40))
        try:
            for seed in seeds:
                for speed in speeds:
                    for scenario in scenarios:
                        for profile in profiles:
                            result,_=run_one(env,map_name,seed,speed,scenario,a.gap,profile,
                                      pdd,steering,old,new,a.steps,a.trace_stride)
                            report['episodes'].append(result)
                            s=result['summary']
                            print(map_name,seed,speed,scenario,profile,s['status'],
                               'min_gap=',s['min_front_gap_m'],flush=True)
        finally:env.close()
    report['success_by_profile']={profile:{'arrive':sum(e['profile']==profile and
        e['summary']['status']=='arrive_dest' for e in report['episodes']),
        'total':sum(e['profile']==profile for e in report['episodes']),
        'crash':sum(e['profile']==profile and e['summary']['status']=='crash' for e in report['episodes'])}
        for profile in profiles}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('SCORES',report['success_by_profile'],'Saved:',a.out.resolve())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',required=True,type=Path);p.add_argument('--steering',required=True,type=Path)
    p.add_argument('--old-longitudinal',required=True,type=Path)
    p.add_argument('--longitudinal',required=True,type=Path)
    p.add_argument('--maps',default='S');p.add_argument('--seeds',default='116,117')
    p.add_argument('--speeds',default='2,3,4');p.add_argument('--scenarios',default=','.join(SCENARIOS))
    p.add_argument('--profiles',default=','.join(PROFILES))
    p.add_argument('--gap',type=float,default=24.);p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--trace-stride',type=int,default=30)
    p.add_argument('--out',type=Path,default=Path('agent4_v38_traffic_eval.json'))
    a=p.parse_args()
    if not 1<=a.trace_stride<=100 or not 100<=a.steps<=3000:p.error('Bad trace-stride or steps')
    try:evaluate(a)
    except ValueError as e:p.error(str(e))
if __name__=='__main__':main()
