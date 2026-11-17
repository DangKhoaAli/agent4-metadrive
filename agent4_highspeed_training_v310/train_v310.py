"""Fit V3.10 learned smooth steering residual from successful teacher rollouts.

No MetaDrive required for training. Learner predicts residual without any
rule-based correction at evaluation. Teacher rule is solely a label source.
Evaluate the model vs *zero correction* (PDD baseline) on validation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from common_v310 import (VERSION,DATA_STATUS,MAX_STEER,MAX_RESIDUAL,BASIS_NAMES,
    basis,check_pdd,feature_names,finite,clip,validate_seed_groups)


def read_data(path,names):
    required=set(names)|{'map','seed','target_speed_mps','teacher_steer_md',
                          'noise_md','pdd_steer_md','step'}
    x=[];y=[];bases=[];maps=[];seeds=[];speeds=[];noises=[];steps=[]
    with path.open(newline='',encoding='utf-8') as file:
        reader=csv.DictReader(file)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing data fields {sorted(required-set(reader.fieldnames or []))}')
        for n,row in enumerate(reader):
            try:
                state={k:finite(row[k],f'{n}:{k}') for k in names}
                base=state['base_pdd_steer_md']
                if abs(base-finite(row['pdd_steer_md']))>1e-6:
                    raise ValueError('PDD base mismatch')
                x.append(basis(state['planned_lateral_right_m'],
                               state['heading_error_rad'],state['speed_mps']))
                y.append(clip(finite(row['teacher_steer_md'])-base,MAX_RESIDUAL))
                bases.append(base);maps.append(row['map']);seeds.append(int(row['seed']))
                speeds.append(float(row['target_speed_mps']))
                noises.append(float(row['noise_md']));steps.append(int(row['step']))
            except Exception as exc:
                raise ValueError(f'{path}:{n+2}: {exc}') from exc
    if not x:raise ValueError(f'Empty dataset: {path}')
    return {'x':np.vstack(x),'y':np.asarray(y,dtype=float),
            'base':np.asarray(bases,dtype=float),'maps':maps,'seeds':seeds,
            'speeds':speeds,'noises':np.asarray(noises),'steps':steps}


def validate_data(manifest,train,val):
    if manifest.get('status')!=DATA_STATUS:raise ValueError('Wrong V3.10 manifest')
    split=manifest.get('splits',{})
    validated=validate_seed_groups(split['train'],split['val'],split['eval'])
    allowed_maps=set(manifest['maps']);allowed_speed=set(manifest['speeds_mps'])
    for name,data in (('train',train),('val',val)):
        seeds=set(data['seeds'])
        if not seeds<=set(validated[name]):raise ValueError(f'{name}: seed outside declared split')
        if set(data['maps'])-allowed_maps or set(data['speeds'])-allowed_speed:
            raise ValueError(f'{name}: map/speed outside manifest')
        if len(data['y'])!=manifest['rows_by_split'][name]:
            raise ValueError(f'{name}: CSV row count != manifest')
        for m in allowed_maps:
            for speed in allowed_speed:
                indices=[i for i,(mm,vv) in enumerate(zip(data['maps'],data['speeds']))
                         if mm==m and vv==speed]
                need=100 if name=='train' else 50
                if len(indices)<need:
                    raise ValueError(f'{name}: missing successful teacher episodes for {m}/{speed:g}: {len(indices)} rows (need {need}). Inspect manifest first.')
    if set(train['seeds']) & set(val['seeds']):raise ValueError('Seed leakage')
    return True


def sample_weights(data):
    cells=list(zip(data['maps'],data['speeds']))
    c=Counter(cells);n=len(cells)
    # Equal total weight per map-speed group; emphasize actual recovery states.
    e=np.abs(data['x'][:,0]);h=np.abs(data['x'][:,1])
    w=np.asarray([n/len(c)/c[k] for k in cells],dtype=float)
    w*=1.+2.*np.minimum(3.,e/.05+h/.015)
    w*=n/w.sum()
    return w


def metrics(y,pred):
    return {'n':int(len(y)),'mae':float(np.mean(np.abs(y-pred))),
            'rmse':float(np.sqrt(np.mean((y-pred)**2))),
            'max_abs_error':float(np.max(np.abs(y-pred)))}


def describe(model,data):
    base=data['base'];y=data['y'];pred=np.clip(model.predict(data['x']),-MAX_RESIDUAL,MAX_RESIDUAL)
    labels=np.clip(base+y,-MAX_STEER,MAX_STEER)
    student=np.clip(base+pred,-MAX_STEER,MAX_STEER)
    pdd=np.clip(base,-MAX_STEER,MAX_STEER)
    all_stats={'teacher_vs_pdd':metrics(labels,pdd),'teacher_vs_student':metrics(labels,student),
               'residual':metrics(y,pred),'n_perturbed_rows':int(np.sum(np.abs(data['noises'])>1e-12))}
    by_cell={}
    for m,s in sorted(set(zip(data['maps'],data['speeds']))):
        mask=np.asarray([mm==m and vv==s for mm,vv in zip(data['maps'],data['speeds'])])
        by_cell[f'{m}:{s:g}']={'pdd':metrics(labels[mask],pdd[mask]),
                             'student':metrics(labels[mask],student[mask]),
                             'n':int(mask.sum())}
    all_stats['by_cell']=by_cell
    return all_stats


def train(data_dir,pdd_names,alpha=.0001):
    manifest=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    names=feature_names(pdd_names)
    tr=read_data(data_dir/'train.csv',names)
    va=read_data(data_dir/'val.csv',names)
    validate_data(manifest,tr,va)
    if not math.isfinite(alpha) or not 0<=alpha<=10:raise ValueError('Invalid ridge alpha')
    model=Ridge(alpha=alpha,fit_intercept=False,solver='svd')
    model.fit(tr['x'],tr['y'],sample_weight=sample_weights(tr))
    stats={'train':describe(model,tr),'val':describe(model,va),
           'basis_coefficients':{name:float(w) for name,w in zip(BASIS_NAMES,model.coef_)},
           'recovery_coverage':{
              split:{'off_center_2p5cm':int(np.sum(np.abs(data['x'][:,0])>.025)),
                     'off_center_5cm':int(np.sum(np.abs(data['x'][:,0])>.05)),
                     'heading_over_0p01rad':int(np.sum(np.abs(data['x'][:,1])>.01))}
              for split,data in (('train',tr),('val',va))}}
    return {'version':VERSION,'student_model':model,
            'feature_order':names,'original_pdd_features':list(pdd_names),
            'basis_names':BASIS_NAMES,'max_steer':MAX_STEER,
            'max_learned_residual':MAX_RESIDUAL,
            'train_seeds':manifest['splits']['train'],
            'val_seeds':manifest['splits']['val'],
            'eval_seeds_reserved':manifest['splits']['eval'],
            'speed_targets_mps':manifest['speeds_mps'],
            'maps':manifest['maps'],'supervision':'MetaDrive PDD+rule_lateral teacher; not IDM; learner is Ridge zero-centered feedback model',
            'metrics':stats,'limitations':['Model MAE is NOT closed-loop performance.',
                'No teacher steering at inference, governor remains rule-based.',
                '10 m/s only on straight road; no validated C/X curves or traffic.',
                'Privileged MetaDrive geometry; not real-world vehicle control.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdd',required=True,type=Path)
    parser.add_argument('--data-dir',type=Path,default=Path('agent4_v310_steering_data'))
    parser.add_argument('--out-model',type=Path,default=Path('agent4_v310_steering.joblib'))
    parser.add_argument('--out-metrics',type=Path,default=Path('agent4_v310_training_metrics.json'))
    parser.add_argument('--alpha',type=float,default=.0001)
    a=parser.parse_args(argv)
    pdd_names=check_pdd(joblib.load(a.pdd))
    b=train(a.data_dir,pdd_names,alpha=a.alpha)
    a.out_model.parent.mkdir(parents=True,exist_ok=True)
    a.out_metrics.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(b,a.out_model)
    result={'status':'V310_OFFLINE_TEACHER_IMITATION_NOT_DRIVING_EVALUATION',
            'bundle_version':VERSION,'data_dir':str(a.data_dir),
            'models':{'student':str(a.out_model)},'metrics':b['metrics'],
            'seed_splits':{'train':b['train_seeds'],'val':b['val_seeds'],'eval':b['eval_seeds_reserved']},
            'limitations':b['limitations']}
    a.out_metrics.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Train/VAL steering MAE vs teacher:',
          b['metrics']['val']['teacher_vs_pdd']['mae'],
          b['metrics']['val']['teacher_vs_student']['mae'])
    print('RECOVERY coverage:',b['metrics']['recovery_coverage'])
    print('Saved:',a.out_model.resolve(),a.out_metrics.resolve())
    print('Next: closed-loop held-out eval. Do NOT infer autonomous success from offline MAE.')
    return b

if __name__=='__main__':main()
