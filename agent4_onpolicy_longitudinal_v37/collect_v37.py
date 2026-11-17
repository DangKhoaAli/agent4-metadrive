"""Collect executed governor actions under frozen V3.5 learned steering.

Teacher = V3.0/V3.5 speed governor (NOT IDM); data are on-policy with
steering = PDD + learned V3.5 residual. Only successful teacher episodes
are retained. Labels are aligned with the PRE-step state and executed action.
"""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import joblib

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_multispeed_v35'))
from evaluate_multispeed_v35 import check_student
sys.path.insert(0,str(ROOT/'agent4_robustness_v34'))
from benchmark_v34 import v30,eval_one,parse_seeds,parse_speeds,parse_simple_csv
from common_v37 import names,make_row,validate_new_splits,DATA_STATUS

META=['map','seed','step','base_speed_mps','teacher_source','teacher_longitudinal',
      'applied_longitudinal','longitudinal_clipped','lane_class']

@contextmanager
def capture_governor_teacher(pdd_features,records):
    real_f=v30.feature_for_model
    real_geom=v30.planned_geometry
    real_long=v30.governed_speed_action
    cache={}
    def observe(env,max_gap):
        feat,route=real_f(env,max_gap)
        cache['features']=feat
        cache['route']=route
        cache['lane_class']=type(env.agent.lane).__name__ if env.agent.lane is not None else 'None'
        return feat,route
    def geometry(car,max_gap):
        g=real_geom(car,max_gap)
        cache['geometry']=g
        return g
    def action(speed,base_speed,remaining):
        teacher,target=real_long(speed,base_speed,remaining)
        if 'features' not in cache or 'geometry' not in cache or 'route' not in cache:
            raise RuntimeError('Pre-step feature/geometry not captured before teacher action')
        row=make_row(cache['features'],pdd_features,cache['geometry'],
                     cache['route']['adapted'],target,cache['route']['remaining_m'])
        item={'step':len(records),'base_speed_mps':float(base_speed),
              'teacher_source':'v35_governor_under_v35_steering',
              'teacher_longitudinal':float(teacher),'applied_longitudinal':float(teacher),
              'longitudinal_clipped':0,'lane_class':cache['lane_class'],**row}
        records.append(item)
        return float(teacher),float(target)
    v30.feature_for_model=observe
    v30.planned_geometry=geometry
    v30.governed_speed_action=action
    try:yield
    finally:
        v30.feature_for_model=real_f
        v30.planned_geometry=real_geom
        v30.governed_speed_action=real_long


def one(env,map_name,seed,speed,pdd,steering,steps):
    rows=[]
    with capture_governor_teacher(pdd['features'],rows):
        episode=eval_one(env,map_name,seed,speed,'student_v33',pdd,steering,steps,50)
    n=episode['summary']['n_applied_steps']
    if len(rows)!=n:
        raise RuntimeError(f'Teacher/actions misaligned: {len(rows)} recorded vs {n} applied')
    if episode['summary'].get('student_predictions_aligned') is not True:
        raise RuntimeError('V3.5 steering predictions are not action-aligned')
    for t in episode['trace']:
        idx=t['step']
        if abs(rows[idx]['applied_longitudinal'] - t['action_applied'][1])>1e-8:
            raise RuntimeError(f'Teacher longitudinal differs from executed action at {idx}')
    info={'map':map_name,'seed':seed,'base_speed_mps':speed,
          'status':episode['summary']['status'],'n_steps':n,
          'rows_accepted':n if episode['summary']['status']=='arrive_dest' else 0,
          'brake_actions':sum(r['teacher_longitudinal'] < -.02 for r in rows),
          'throttle_actions':sum(r['teacher_longitudinal'] > .02 for r in rows),
          'student_predictions_aligned':True,
          'error_count':episode['summary']['error_count'],
          'max_abs_lateral_m':episode['summary']['max_abs_planned_lateral_right_m']}
    return rows,info


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pdd',type=Path,required=True)
    ap.add_argument('--steering',type=Path,required=True)
    ap.add_argument('--maps',default='S,C,X')
    ap.add_argument('--speeds',default='2,3,4')
    ap.add_argument('--train-seeds',default='80-91')
    ap.add_argument('--val-seeds',default='92-95')
    ap.add_argument('--eval-seeds',default='96-99')
    ap.add_argument('--steps',type=int,default=1800)
    ap.add_argument('--out-dir',type=Path,default=Path('agent4_v37_governor_data'))
    args=ap.parse_args(argv)
    try:
        maps=parse_simple_csv(args.maps,('S','C','X'))
        speeds=parse_speeds(args.speeds)
        splits=validate_new_splits({'train':parse_seeds(args.train_seeds),
             'val':parse_seeds(args.val_seeds),'eval':parse_seeds(args.eval_seeds)})
        if not 100<=args.steps<=5000:raise ValueError('steps outside 100..5000')
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise ValueError('Output directory already has data; use a NEW --out-dir')
    except ValueError as exc:ap.error(str(exc))
    pdd=joblib.load(args.pdd)
    steering=joblib.load(args.steering)
    if list(pdd.get('features',[]))!=list(v30.FEATURES) or pdd.get('label_mode')!='at_t':
        raise ValueError('Expected frozen PDD V2 at_t 15-feature model')
    check_student(steering,pdd['features'],splits['train']+splits['val']+splits['eval'],
                  'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL')
    from metadrive.envs.metadrive_env import MetaDriveEnv
    fields=META+names(pdd['features'])
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'status':DATA_STATUS,'teacher':'rule-based V3.5 speed governor',
              'executed_steering':'frozen PDD V2 + learned V3.5 residual',
              'data_contract':'teacher_longitudinal == actually_applied_longitudinal',
              'maps':maps,'speeds_mps':speeds,'splits':splits,'data_fields':fields,
              'traffic_density':0.0,'episodes':[],
              'rows_by_split':{'train':0,'val':0},'rows_by_speed_map':{},
              'only_successful_teacher_episodes_retained':True,
              'warnings':['Teacher is governor, NOT IDM. This is behavior cloning of a rule-based expert.',
                          'Final speed planning is still hand-coded.',
                          'Zero traffic, privileged simulator features.',
                          'Only successful teacher episodes retained; rare failure states may be underrepresented.']}
    with (args.out_dir/'train.csv').open('w',newline='',encoding='utf8') as ft,\
         (args.out_dir/'val.csv').open('w',newline='',encoding='utf8') as fv:
        wr={'train':csv.DictWriter(ft,fieldnames=fields),
            'val':csv.DictWriter(fv,fieldnames=fields)}
        for writer in wr.values():writer.writeheader()
        for map_name in maps:
            env=None
            try:
                env=MetaDriveEnv(dict(use_render=False,map=map_name,
                    num_scenarios=max(splits['train']+splits['val'])+1,
                    traffic_density=0.0,horizon=max(args.steps+10,1000),log_level=40))
                for speed in speeds:
                    for split in ('train','val'):
                        for seed in splits[split]:
                            try:
                                rows,e=one(env,map_name,seed,speed,pdd,steering,args.steps)
                                if e['status']=='arrive_dest':
                                    for row in rows:wr[split].writerow({'map':map_name,'seed':seed,**row})
                                    manifest['rows_by_split'][split]+=len(rows)
                                    k=f'{split}:{speed:g}:{map_name}'
                                    manifest['rows_by_speed_map'][k]=len(rows)+manifest['rows_by_speed_map'].get(k,0)
                            except Exception as ex:
                                e={'map':map_name,'seed':seed,'base_speed_mps':speed,
                                   'status':'collection_error','rows_accepted':0,
                                   'errors':[f'{type(ex).__name__}: {ex}']}
                            e['split']=split
                            manifest['episodes'].append(e)
                            print(f'{map_name} {speed:g}m/s {split} seed={seed}: {e["status"]} rows={e["rows_accepted"]}',flush=True)
            finally:
                if env is not None:env.close()
    manifest['success_by_split']={split:sum(e['split']==split and e['status']=='arrive_dest' for e in manifest['episodes'])
         for split in ('train','val')}
    manifest['accepted_brake_steps']=sum(e.get('brake_actions',0) for e in manifest['episodes'] if e['status']=='arrive_dest')
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('Manifest saved:',(args.out_dir/'manifest.json').resolve(),'rows',manifest['rows_by_split'])
    if min(manifest['rows_by_split'].values())<500:
        raise RuntimeError('Insufficient successful samples; inspect manifest for failures')
    if manifest['accepted_brake_steps']<30:
        raise RuntimeError('Insufficient brake labels; inspect manifest')

if __name__=='__main__':main()
