"""V3.5: orchestrate V3.3 IDM teacher collection at several target speeds.

Each speed is collected with the same disjoint seed split, and only successful
IDM episodes contribute teacher rows. Dataset includes observed speed in the
original PDD features; target_speed_mps is for audit/stratified metrics only.
Research simulation only; MetaDrive IDM teacher runs ONLY during collection.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_robustness_v34'))
from benchmark_v34 import parse_seeds,parse_speeds,parse_simple_csv

TEACHER=ROOT/'agent4_learned_residual_v33'/'collect_teacher_v33.py'


def validate_groups(train,val,heldout):
    for label,group in [('train',train),('val',val),('eval',heldout)]:
        if not group or min(group)<0 or len(group)!=len(set(group)):
            raise ValueError(f'Invalid {label} seed set')
    if set(train)&set(val) or set(train)&set(heldout) or set(val)&set(heldout):
        raise ValueError('Train/validation/evaluation seed leakage')
    if set(train+val)&set(range(0,40)):
        raise ValueError('Seeds 0..39 were used in prior audits/evaluations: never train on them')


def speed_folder(speed):
    return 'speed_'+format(speed,'.3f').replace('.','p')


def merge_collected(data_dir,speeds,train_seeds,val_seeds,eval_seeds,maps,perturb):
    data_dir=Path(data_dir)
    manifests=[]
    output_fields=None
    rows_by_split={'train':0,'val':0}
    rows_by_speed_map={}
    episode_rows=[]
    sources={}
    for speed in speeds:
        directory=data_dir/speed_folder(speed)
        mp=directory/'manifest.json'
        if not mp.is_file():raise FileNotFoundError(f'Missing teacher manifest: {mp}')
        d=json.loads(mp.read_text(encoding='utf-8'))
        if d.get('splits') != {'train':train_seeds,'val':val_seeds,'eval':eval_seeds}:
            raise ValueError(f'Seed split mismatch in {mp}')
        if d.get('maps')!=maps or not math.isclose(float(d.get('speed_target_mps',-1)),speed):
            raise ValueError(f'Map/speed mismatch in {mp}')
        if not math.isclose(float(d.get('perturbation',-1)),perturb):
            raise ValueError(f'Perturbation mismatch in {mp}')
        manifests.append(d)
        for e in d['episodes']:
            episode_rows.append({'target_speed_mps':speed,**e})
        for split in ('train','val'):
            fp=directory/f'{split}.csv'
            with fp.open('r',newline='',encoding='utf-8') as handle:
                reader=csv.DictReader(handle)
                header=reader.fieldnames
                if not header or 'speed_mps' not in header or 'teacher_steer_md' not in header:
                    raise ValueError(f'Invalid teacher CSV header: {fp}')
                if output_fields is None:output_fields=['target_speed_mps']+header
                if output_fields!=['target_speed_mps']+header:
                    raise ValueError(f'Different feature order at {fp}')
                n=0
                for row in reader:
                    if int(row['seed']) not in (train_seeds if split=='train' else val_seeds):
                        raise ValueError(f'Leaked seed in {fp}')
                    if row['map'] not in maps:raise ValueError(f'Invalid map in {fp}')
                    n+=1
                if n!=d['rows_by_split'][split]:
                    raise ValueError(f'CSV/manifest row mismatch {fp}: {n} != {d["rows_by_split"][split]}')
                sources[(speed,split)]=fp
                rows_by_split[split]+=n
    if output_fields is None:raise ValueError('No columns')
    for split in ('train','val'):
        target=data_dir/f'{split}.csv'
        with target.open('w',newline='',encoding='utf-8') as out:
            wr=csv.DictWriter(out,fieldnames=output_fields)
            wr.writeheader()
            for speed in speeds:
                source=sources[(speed,split)]
                with source.open('r',newline='',encoding='utf-8') as handle:
                    for row in csv.DictReader(handle):
                        enriched={'target_speed_mps':speed,**row}
                        wr.writerow(enriched)
                        key=f'{split}:{speed:g}:{row["map"]}'
                        rows_by_speed_map[key]=rows_by_speed_map.get(key,0)+1
    # 4m/s teacher can fail on some routes; make the problem explicit in manifest.
    success_counts={}
    for speed in speeds:
        for split in ('train','val'):
            for name in maps:
                subset=[e for e in episode_rows if e['target_speed_mps']==speed and e.get('split')==split and e.get('map')==name]
                success_counts[f'{split}:{speed:g}:{name}']={
                    'success':sum(e.get('status')=='arrive_dest' for e in subset),
                    'total':len(subset),
                    'retained_rows':rows_by_speed_map.get(f'{split}:{speed:g}:{name}',0)}
    manifest={'status':'V35_MULTISPEED_IDM_STEERING_TEACHER_DATA',
              'maps':maps,'speeds_mps':speeds,
              'splits':{'train':train_seeds,'val':val_seeds,'eval':eval_seeds},
              'perturbation':perturb,'data_fields':output_fields,
              'rows_by_split':rows_by_split,'rows_by_speed_map':rows_by_speed_map,
              'success_by_split_speed_map':success_counts,'episodes':episode_rows,
              'supervision':'MetaDrive IDM steering only; rule-based governor for longitudinal',
              'only_successful_teacher_episodes_kept':True,
              'limitations':['Observed speed_mps is an existing model feature; target_speed_mps is audit metadata only.',
                             'No control of longitudinal by a learned model.',
                             'Never train on seeds 0..39, reserved eval seeds, or any closed-loop evaluation seeds.',
                             'All training/eval uses privileged simulator road geometry and zero traffic.']}
    (data_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--maps',default='S,C,X')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--train-seeds',default='40-51')
    p.add_argument('--val-seeds',default='52-55')
    p.add_argument('--eval-seeds',default='56-59',help='Reserved: NEVER collected')
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--perturb-md',type=float,default=0.008)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v35_multispeed_data'))
    p.add_argument('--reuse-existing',action='store_true',help='Reuse a complete per-speed dataset after strict manifest checks')
    a=p.parse_args(argv)
    try:
        maps=parse_simple_csv(a.maps,('S','C','X'))
        speeds=parse_speeds(a.speeds)
        train,val,heldout=[parse_seeds(v) for v in (a.train_seeds,a.val_seeds,a.eval_seeds)]
        validate_groups(train,val,heldout)
        if a.steps<100 or a.steps>4000:raise ValueError('steps outside 100..4000')
        if not math.isfinite(a.perturb_md) or not 0<=a.perturb_md<=0.035:
            raise ValueError('perturb-md outside 0..0.035')
    except ValueError as exc:p.error(str(exc))
    if not TEACHER.is_file():raise FileNotFoundError(TEACHER)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for speed in speeds:
        folder=a.out_dir/speed_folder(speed)
        if (folder/'manifest.json').exists() and a.reuse_existing:
            print('REUSE existing speed dataset (validated during merge):',folder,flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'Nonempty folder {folder}. Choose new --out-dir or use --reuse-existing.')
        cmd=[sys.executable,str(TEACHER),'--model',str(a.model),
             '--maps',','.join(maps),'--train-seeds',','.join(map(str,train)),
             '--val-seeds',','.join(map(str,val)),
             '--eval-seeds',','.join(map(str,heldout)),
             '--steps',str(a.steps),'--speed-mps',str(speed),
             '--perturb-md',str(a.perturb_md),'--out-dir',str(folder)]
        print('COLLECT teacher speed',speed,'m/s',flush=True)
        subprocess.run(cmd,check=True)
    report=merge_collected(a.out_dir,speeds,train,val,heldout,maps,a.perturb_md)
    print('Merged teacher data:',a.out_dir.resolve(),flush=True)
    print('Rows by split:',report['rows_by_split'])
    print('Teacher coverage:')
    for name,value in report['success_by_split_speed_map'].items():
        print(' ',name,value)
    print('Inspect success_by_split_speed_map before training. IDM success is not Agent 4 success.')


if __name__=='__main__':main()
