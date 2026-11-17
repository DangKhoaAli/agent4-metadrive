"""Train V3.3 model-only steering residual from MetaDrive IDM teacher CSV.

Train and validation are separated by complete (map,seed) episodes. Holdout
seeds are never used in either. The PDD V2 model stays frozen in the adapter.
Metric here is TEACHER IMITATION, not autonomous driving success.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common_v33 import MAX_STEER,MAX_STUDENT_RESIDUAL,finite,clip,feature_names


def read_csv(path,names):
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if not set(names+['teacher_steer_md','map','seed']).issubset(reader.fieldnames or []):
            raise ValueError(f'Missing required columns in {path}')
        rows=[]
        for i,r in enumerate(reader):
            values=[finite(r[k],f'{path}:{i}:{k}') for k in names]
            teacher=clip(finite(r['teacher_steer_md']),MAX_STEER)
            rows.append((np.asarray(values,dtype=float),teacher,r['map'],int(r['seed'])))
    if not rows: raise ValueError(f'Empty dataset {path}')
    x=np.vstack([z[0] for z in rows]); y=np.asarray([z[1] for z in rows])
    return x,y,[z[2] for z in rows],[z[3] for z in rows]


def eval_metrics(y_true,pred):
    return {'n':int(len(y_true)),
            'steering_mae':float(mean_absolute_error(y_true,pred)),
            'steering_rmse':float(np.sqrt(mean_squared_error(y_true,pred)))}


def assemble_predictions(estimator,x,base_index):
    residual=np.clip(np.asarray(estimator.predict(x),dtype=float),
                     -MAX_STUDENT_RESIDUAL,MAX_STUDENT_RESIDUAL)
    return np.clip(x[:,base_index]+residual,-MAX_STEER,MAX_STEER)


def train(train_path,val_path,pdd_names,manifest_path=None,*,max_iter=220):
    names=feature_names(pdd_names)
    tr_x,tr_y,tr_maps,tr_seeds=read_csv(train_path,names)
    va_x,va_y,va_maps,va_seeds=read_csv(val_path,names)
    gtrain=set(zip(tr_maps,tr_seeds)); gval=set(zip(va_maps,va_seeds))
    if gtrain & gval:raise ValueError(f'TRAIN/VAL episode leakage: {gtrain & gval}')
    if len(tr_y)<500 or len(va_y)<100:
        raise ValueError('Need at least 500 train and 100 val rows; collect additional successful teacher episodes')
    if manifest_path is not None:
        manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        train_ids=set(manifest['splits']['train']);val_ids=set(manifest['splits']['val']);ev_ids=set(manifest['splits']['eval'])
        if set(tr_seeds)-train_ids or set(va_seeds)-val_ids:
            raise ValueError('CSV seed is outside its manifest split')
        if ev_ids & (set(tr_seeds)|set(va_seeds)):
            raise ValueError('Reserved evaluation seeds leaked into train or val')
    idx=names.index('base_pdd_steer_md')
    tr_delta=np.clip(tr_y-tr_x[:,idx],-MAX_STUDENT_RESIDUAL,MAX_STUDENT_RESIDUAL)
    model=HistGradientBoostingRegressor(
        loss='squared_error',learning_rate=0.065,max_iter=int(max_iter),
        max_leaf_nodes=18,min_samples_leaf=35,l2_regularization=2.0,
        early_stopping=False,random_state=42)
    model.fit(tr_x,tr_delta)
    results={}
    for split,x,y,maps,seeds_ in (('train',tr_x,tr_y,tr_maps,tr_seeds),
                                  ('val',va_x,va_y,va_maps,va_seeds)):
        base=np.clip(x[:,idx],-MAX_STEER,MAX_STEER)
        pred=assemble_predictions(model,x,idx)
        metrics={'baseline_pdd':eval_metrics(y,base),
                 'learned_student':eval_metrics(y,pred),
                 'student_better_mae':bool(mean_absolute_error(y,pred)<mean_absolute_error(y,base)),
                 'n_episodes':len(set(zip(maps,seeds_))),
                 'by_map':{}}
        for m in sorted(set(maps)):
            ids=np.array([v==m for v in maps])
            metrics['by_map'][m]={'baseline_pdd':eval_metrics(y[ids],base[ids]),
                                  'learned_student':eval_metrics(y[ids],pred[ids])}
        results[split]=metrics
    bundle={'version':'V33_META_DRIVE_IDM_DISTILLED_RESIDUAL',
            'student_model':model,'feature_order':names,'original_pdd_features':list(pdd_names),
            'max_steer':MAX_STEER,'max_learned_residual':MAX_STUDENT_RESIDUAL,
            'fixed_pdd_model_required':True,
            'supervision':'MetaDrive IDM steering at 3 m/s, NOT original CARLA/PDD labels',
            'train_seed_groups':sorted(gtrain),'val_seed_groups':sorted(gval),
            'metrics':results,
            'limitations':['Teacher imitation MAE is not closed-loop task success.',
                           'MetaDrive privileged geometry features; traffic-free simulation only.',
                           'Training from near-expert trajectories does not guarantee recovery from large errors.',
                           'Original PDD longitudinal model is not deployed.']}
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('agent4_v33_teacher_data'))
    p.add_argument('--model',type=Path,required=True,help='Original PDD V2 joblib for immutable feature order')
    p.add_argument('--out-model',type=Path,default=Path('agent4_v33_student.joblib'))
    p.add_argument('--out-metrics',type=Path,default=Path('agent4_v33_training_metrics.json'))
    p.add_argument('--max-iter',type=int,default=220)
    a=p.parse_args()
    if not 20<=a.max_iter<=1500:p.error('--max-iter must be 20..1500')
    pdd=joblib.load(a.model)
    names=list(pdd['features'])
    if pdd.get('label_mode')!='at_t' or len(names)!=15:
        raise ValueError('Expected original frozen CARLA PDD model V2')
    bundle=train(a.data_dir/'train.csv',a.data_dir/'val.csv',names,
                 a.data_dir/'manifest.json',max_iter=a.max_iter)
    a.out_model.parent.mkdir(parents=True,exist_ok=True)
    a.out_metrics.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,a.out_model)
    report={'status':'V33_TEACHER_IMITATION_METRICS_NOT_CLOSED_LOOP',
            'model_path':str(a.out_model),'supervision':bundle['supervision'],
            'features':bundle['feature_order'],
            'metrics':bundle['metrics'],
            'train_episodes':len(bundle['train_seed_groups']),
            'val_episodes':len(bundle['val_seed_groups']),
            'limits':bundle['limitations']}
    a.out_metrics.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for split in ('train','val'):
        r=bundle['metrics'][split]
        print(f'{split}: n={r["learned_student"]["n"]} '
              f'PDD_MAE={r["baseline_pdd"]["steering_mae"]:.5f} '
              f'Student_MAE={r["learned_student"]["steering_mae"]:.5f} '
              f'better={r["student_better_mae"]}')
    print('Saved student model:',a.out_model.resolve())
    print('Saved offline metrics:',a.out_metrics.resolve())
    print('Caution: offline teacher action accuracy is NOT driving success.')

if __name__=='__main__':main()
