"""A/B/C closed-loop: frozen V3.5 steering with governor, V3.6 IDM student, V3.7 governor student.

V3.7 learned action DIRECTLY applies (no teacher fallback). Only arrive_dest=success.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_longitudinal_v36'))
import eval_longitudinal_v36 as v36
from common_v37 import VERSION,names

PROFILES=('governor_v35','learned_long_v36','learned_governor_v37')

def validate_models(pdd,steering,old,new,seeds,speeds):
    if pdd.get('label_mode')!='at_t' or len(pdd.get('features',[]))!=15:
        raise ValueError('Expected PDD V2 at_t 15 features')
    v36.check_student(steering,pdd['features'],seeds,'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    for model,version in [(old,v36.VERSION),(new,VERSION)]:
        if model.get('version')!=version or model.get('pdd_features')!=pdd['features'] or model.get('feature_order')!=names(pdd['features']):
            raise ValueError(f'Longitudinal model version/features incompatible: {model.get("version")}')
        v36.check_eval_seeds(model,seeds)
        if not set(speeds)<=set(model.get('speed_targets',[])):
            raise ValueError('Requested speed was not in training speed targets')
    if min(seeds)<80:raise ValueError('Use new V3.7 seeds >=80; seed 76/77 are diagnostics')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pdd',type=Path,required=True)
    ap.add_argument('--steering',type=Path,required=True)
    ap.add_argument('--old-longitudinal',type=Path,required=True)
    ap.add_argument('--longitudinal',type=Path,required=True)
    ap.add_argument('--maps',default='C,X')
    ap.add_argument('--seeds',default='96,97')
    ap.add_argument('--speeds',default='2,3,4')
    ap.add_argument('--profiles',default=','.join(PROFILES))
    ap.add_argument('--steps',type=int,default=1800)
    ap.add_argument('--trace-stride',type=int,default=20)
    ap.add_argument('--out',type=Path,default=Path('agent4_v37_closed_loop_eval.json'))
    a=ap.parse_args(argv)
    try:
        maps=v36.parse_simple_csv(a.maps,('S','C','X'))
        seeds=v36.parse_seeds(a.seeds)
        speeds=v36.parse_speeds(a.speeds)
        profiles=v36.parse_simple_csv(a.profiles,PROFILES)
        if not 100<=a.steps<=5000 or not 1<=a.trace_stride<=100:
            raise ValueError('Bad steps/trace stride')
    except ValueError as exc:ap.error(str(exc))
    pdd=joblib.load(a.pdd)
    steering=joblib.load(a.steering)
    old=joblib.load(a.old_longitudinal)
    new=joblib.load(a.longitudinal)
    validate_models(pdd,steering,old,new,seeds,speeds)
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report={'status':'V37_CLOSED_LOOP_GOVERNOR_IMITATION_AB_C',
            'maps':maps,'seeds':seeds,'speeds_mps':speeds,'profiles':profiles,
            'frozen_steering_v35':True,'learned_v37_no_governor_fallback':True,
            'traffic_density':0.0,'idm_used_for_control':False,
            'warnings':['Teacher for V3.7 is GOVERNOR, not IDM.',
                        'Only arrive_dest counts as success.',
                        'The target speed planner remains rule-based.',
                        'V3.7 learned mode predicts signed longitudinal directly.',
                        'No traffic; privileged MetaDrive geometry; not physical safety evidence.'],
            'episodes':[]}
    for m in maps:
        env=None
        try:
            env=MetaDriveEnv(dict(use_render=False,map=m,num_scenarios=max(seeds)+1,
                     traffic_density=0.0,horizon=max(1000,a.steps+10),log_level=40))
            for seed in seeds:
                for speed in speeds:
                    for profile in profiles:
                        internal='governor_v35' if profile=='governor_v35' else 'learned_long_v36'
                        b=old if profile=='learned_long_v36' else new
                        item=v36.one(env,m,seed,speed,internal,pdd,steering,b,a.steps,a.trace_stride)
                        item['profile']=profile
                        item['longitudinal_teacher_source']=(None if profile=='governor_v35' else
                            'native_IDM' if profile=='learned_long_v36' else 'V3.5_governor')
                        report['episodes'].append(item)
                        print(m,seed,speed,profile,item['summary']['status'],
                              round(item['summary']['travelled_approx_m'],2),flush=True)
        finally:
            if env is not None:env.close()
    report['success_by_profile']={p:{'success':sum(e['profile']==p and e['summary']['status']=='arrive_dest' for e in report['episodes']),
        'total':sum(e['profile']==p for e in report['episodes'])} for p in profiles}
    report['scores_by_speed_map_profile']=v36.group_scores(v36.summary_rows(report['episodes']))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('Saved',a.out.resolve(),'success',report['success_by_profile'])

if __name__=='__main__':main()
