"""Collect aligned demonstrations from a headway-aware RULE teacher with scripted lead cars."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_robustness_v34'))
from benchmark_v34 import parse_seeds,parse_speeds,parse_simple_csv,v30
sys.path.insert(0,str(ROOT/'agent4_multispeed_v35'))
from evaluate_multispeed_v35 import check_student
from common_v38 import VERSION,STATUS,SCENARIOS,feature_names,validate_splits
from traffic_v38 import run_one


def collect(args):
    splits=validate_splits(dict(train=parse_seeds(args.train_seeds),
        val=parse_seeds(args.val_seeds),eval=parse_seeds(args.eval_seeds)))
    maps=parse_simple_csv(args.maps,('S',))
    speeds=parse_speeds(args.speeds)
    scenarios=parse_simple_csv(args.scenarios,SCENARIOS)
    if not 100<=args.steps<=3000:raise ValueError('steps outside 100..3000')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('Output already exists; choose a clean directory')
    pdd=joblib.load(args.pdd);steering=joblib.load(args.steering)
    if pdd.get('features')!=list(v30.FEATURES) or pdd.get('label_mode')!='at_t':
        raise ValueError('Wrong PDD bundle')
    check_student(steering,pdd['features'],splits['train']+splits['val']+splits['eval'],
                  'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    features=feature_names(pdd['features'])
    fields=['map','seed','scenario','step','base_speed_mps','teacher_longitudinal',
            'applied_longitudinal','teacher_target_mps','route_target_mps']+features
    manifest=dict(version=VERSION,status=STATUS,teacher='scripted gap-aware rule policy (NOT IDM)',
                  steering='frozen PDD+V3.5',traffic_setup='one explicitly spawned controlled lead',
                  maps=maps,speeds_mps=speeds,scenarios=scenarios,
                  gaps_m=args.gap,splits=splits,data_fields=fields,
                  traffic_density=0.,note='Native random traffic disabled; one scripted lead inserted per reset',
                  accepted_rows={'train':0,'val':0},episodes=[],
                  warnings=['Model learns to imitate a hand-coded following teacher, NOT an IDM traffic expert.',
                            'Test initially on S map only; no lane change or intersection reasoning.',
                            'Only successful episodes with meaningful lead exposure are retained.',
                            'Simulator geometry and all surrounding vehicle states are privileged.'])
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'train.csv').open('w',newline='',encoding='utf8') as ft,\
         (args.out_dir/'val.csv').open('w',newline='',encoding='utf8') as fv:
        writers={'train':csv.DictWriter(ft,fields),'val':csv.DictWriter(fv,fields)}
        for w in writers.values():w.writeheader()
        for map_name in maps:
            env=MetaDriveEnv(dict(use_render=False,map=map_name,
                num_scenarios=max(splits['train']+splits['val'])+1,
                traffic_density=0.,horizon=max(1000,args.steps+10),log_level=40))
            try:
                for sp in speeds:
                    for scenario in scenarios:
                        for split in ('train','val'):
                            for seed in splits[split]:
                                try:
                                    result,records=run_one(env,map_name,seed,sp,scenario,args.gap,
                                          'teacher_v38',pdd,steering,None,None,args.steps,30)
                                    summary=result['summary']
                                    passed=(summary['status']=='arrive_dest' and
                                            summary['front_seen_steps']>=20 and
                                            summary['teacher_intervention_steps']>=5 and
                                            summary['front_close_steps']>=5 and
                                            summary['braking_steps']>=5 and
                                            summary['error_count']==0)
                                    if passed:
                                        for row in records:
                                            writers[split].writerow({'map':map_name,'seed':seed,
                                                'scenario':scenario,'step':row['step'],
                                                'base_speed_mps':sp,
                                                'teacher_longitudinal':row['teacher_action'],
                                                'applied_longitudinal':row['applied_longitudinal'],
                                                'teacher_target_mps':row['teacher_target_mps'],
                                                'route_target_mps':row['route_target_mps'],
                                                **row['row']})
                                        manifest['accepted_rows'][split]+=len(records)
                                    item=dict(map=map_name,seed=seed,scenario=scenario,speed_mps=sp,
                                        status=summary['status'],accepted=passed,
                                        n_steps=summary['n_applied_steps'],
                                        front_seen_steps=summary['front_seen_steps'],
                                        front_close_steps=summary['front_close_steps'],
                                        teacher_intervention_steps=summary['teacher_intervention_steps'],
                                        braking_steps=summary['braking_steps'],
                                        min_front_gap_m=summary['min_front_gap_m'],
                                        error_count=summary['error_count'])
                                except Exception as e:
                                    item=dict(map=map_name,seed=seed,scenario=scenario,speed_mps=sp,
                                      status='collection_error',accepted=False,error=f'{type(e).__name__}: {e}')
                                item['split']=split;manifest['episodes'].append(item)
                                print(f'{map_name} {sp:g} {scenario} {split} {seed}: '
                                      f'{item["status"]} accepted={item["accepted"]}',flush=True)
            finally:env.close()
    manifest['accepted_episodes']={s:sum(e['split']==s and e['accepted'] for e in manifest['episodes'])
                                    for s in ('train','val')}
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf8')
    print('Wrote',args.out_dir/'manifest.json',manifest['accepted_rows'])
    if min(manifest['accepted_rows'].values())<300 or min(manifest['accepted_episodes'].values())<2:
        raise RuntimeError('Insufficient verified traffic data; inspect manifest. '
                           'Do not train without real front-car interactions.')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',type=Path,required=True);p.add_argument('--steering',type=Path,required=True)
    p.add_argument('--maps',default='S');p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--scenarios',default=','.join(SCENARIOS))
    p.add_argument('--train-seeds',default='100-111');p.add_argument('--val-seeds',default='112-115')
    p.add_argument('--eval-seeds',default='116-119');p.add_argument('--gap',type=float,default=24.)
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v38_traffic_data'))
    a=p.parse_args()
    try:collect(a)
    except ValueError as e:p.error(str(e))
if __name__=='__main__':main()
