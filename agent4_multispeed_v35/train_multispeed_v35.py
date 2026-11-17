"""Train V3.5 speed-diverse learned steering residual from IDM teacher samples.

Frozen PDD base; supervised target = bounded (IDM steer - PDD steer).
Balances each (speed,map) cell to avoid long episodes dominating training.
No MetaDrive needed for fitting. Research simulation only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_learned_residual_v33'))
from common_v33 import MAX_STEER,MAX_STUDENT_RESIDUAL,feature_names
from train_student_v33 import read_csv,assemble_predictions,eval_metrics


def read_target_speeds(path):
    with Path(path).open('r',encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if 'target_speed_mps' not in (reader.fieldnames or []):
            raise ValueError('Expected target_speed_mps metadata column')
        return np.asarray([float(row['target_speed_mps']) for row in reader])


def verify_dataset(manifest,tr_maps,tr_seeds,va_maps,va_seeds,tr_speed,va_speed):
    split=manifest['splits']
    t,v,h=[set(split[k]) for k in ('train','val','eval')]
    if t&v or t&h or v&h or min(t|v|h)<0:raise ValueError('Manifest split overlap')
    if t & set(range(40)) or v & set(range(40)):
        raise ValueError('Previous evaluation seeds 0..39 cannot be used for V3.5 training')
    if not set(tr_seeds)<=t or not set(va_seeds)<=v:
        raise ValueError('CSV seed does not match manifest split')
    if (set(tr_seeds)|set(va_seeds))&h:
        raise ValueError('Heldout data leakage')
    groups_tr=set(zip(tr_maps,tr_seeds,tr_speed))
    groups_va=set(zip(va_maps,va_seeds,va_speed))
    if groups_tr&groups_va or set(tr_seeds)&set(va_seeds):
        raise ValueError('Overlapping train/val episodes or seeds')
    observed_tr=set(tr_speed)
    observed_val=set(va_speed)
    declared=set(manifest['speeds_mps'])
    if observed_tr!=declared or observed_val!=declared:
        raise ValueError('Each requested speed needs train and validation data')
    if len(declared)<2:raise ValueError('This experiment requires >=2 distinct speed groups')
    for speed in declared:
        for m in manifest['maps']:
            count_tr=sum(z==speed and n==m for z,n in zip(tr_speed,tr_maps))
            count_va=sum(z==speed and n==m for z,n in zip(va_speed,va_maps))
            if count_tr<100 or count_va<50:
                raise ValueError(f'Insufficient successful teacher data for speed={speed:g}, map={m}: train={count_tr}, val={count_va}')


def equal_speed_map_weights(speeds,maps):
    """Normalize each speed/map cell to equal total weight."""
    groups=list(zip([float(z) for z in speeds],maps))
    counts=Counter(groups)
    if not counts:raise ValueError('Empty dataset')
    w=np.asarray([len(groups)/(len(counts)*counts[group]) for group in groups],dtype=float)
    if not np.all(np.isfinite(w)) or not np.all(w>0):raise ValueError('Invalid weights')
    return w


def group_metrics(true,base,pred,speeds,maps):
    results={}
    for speed in sorted(set(speeds)):
        results[str(speed)]={}
        for name in sorted(set(maps)):
            ids=np.array([float(a)==float(speed) and b==name for a,b in zip(speeds,maps)])
            if not ids.any():continue
            results[str(speed)][name]={
                'original_pdd':eval_metrics(true[ids],base[ids]),
                'student_v35':eval_metrics(true[ids],pred[ids]),
                'improved_mae':bool(mean_absolute_error(true[ids],pred[ids]) <
                                    mean_absolute_error(true[ids],base[ids]))}
    return results


def train(data_dir,pdd_features,max_iter=260):
    data_dir=Path(data_dir)
    manifest=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='V35_MULTISPEED_IDM_STEERING_TEACHER_DATA':
        raise ValueError('Not a V3.5 collected manifest')
    names=feature_names(pdd_features)
    tr_x,tr_y,tr_maps,tr_seed=read_csv(data_dir/'train.csv',names)
    va_x,va_y,va_maps,va_seed=read_csv(data_dir/'val.csv',names)
    tr_speed=read_target_speeds(data_dir/'train.csv')
    va_speed=read_target_speeds(data_dir/'val.csv')
    verify_dataset(manifest,tr_maps,tr_seed,va_maps,va_seed,tr_speed,va_speed)
    base_idx=names.index('base_pdd_steer_md')
    teacher_delta=np.clip(tr_y-tr_x[:,base_idx],-MAX_STUDENT_RESIDUAL,MAX_STUDENT_RESIDUAL)
    weights=equal_speed_map_weights(tr_speed,tr_maps)
    reg=HistGradientBoostingRegressor(
        loss='squared_error',learning_rate=.06,max_iter=int(max_iter),
        max_leaf_nodes=24,min_samples_leaf=30,l2_regularization=2.0,
        early_stopping=False,random_state=42)
    reg.fit(tr_x,teacher_delta,sample_weight=weights)
    metrics={}
    for name,x,y,speed,maps in [('train',tr_x,tr_y,tr_speed,tr_maps),('val',va_x,va_y,va_speed,va_maps)]:
        base=np.clip(x[:,base_idx],-MAX_STEER,MAX_STEER)
        pred=assemble_predictions(reg,x,base_idx)
        metrics[name]={
            'original_pdd':eval_metrics(y,base),
            'student_v35':eval_metrics(y,pred),
            'by_target_speed_and_map':group_metrics(y,base,pred,speed,maps),
            'n':len(y)}
    train_groups=sorted(set((str(m),int(s)) for m,s in zip(tr_maps,tr_seed)))
    val_groups=sorted(set((str(m),int(s)) for m,s in zip(va_maps,va_seed)))
    bundle={
        'version':'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL',
        'student_model':reg,'feature_order':names,
        'original_pdd_features':list(pdd_features),
        'max_steer':MAX_STEER,'max_learned_residual':MAX_STUDENT_RESIDUAL,
        'fixed_pdd_model_required':True,
        'supervision':'MetaDrive IDM steering at target speeds '+','.join(map(str,manifest['speeds_mps'])),
        'train_seed_groups':train_groups,'val_seed_groups':val_groups,
        'train_speed_targets_mps':list(manifest['speeds_mps']),
        'metrics':metrics,
        'limitations':['Longitudinal still rule-based; no traffic; simulator privileged state.',
                       'IDM teacher labels and offline MAE are not closed-loop success.',
                       'Training includes only successful IDM teacher episodes.',
                       'Target_speed_mps is metadata, not a model input; observed speed_mps is the input.']}
    return bundle


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path,help='Frozen PDD V2 bundle')
    p.add_argument('--data-dir',type=Path,default=Path('agent4_v35_multispeed_data'))
    p.add_argument('--out-model',type=Path,default=Path('agent4_v35_student.joblib'))
    p.add_argument('--out-metrics',type=Path,default=Path('agent4_v35_training_metrics.json'))
    p.add_argument('--max-iter',type=int,default=260)
    a=p.parse_args(argv)
    if not 40<=a.max_iter<=1200:p.error('--max-iter must be 40..1200')
    pdd=joblib.load(a.model)
    if pdd.get('label_mode')!='at_t' or len(pdd.get('features',[]))!=15:
        raise ValueError('Expected frozen PDD V2 at_t model with 15 features')
    result=train(a.data_dir,pdd['features'],a.max_iter)
    a.out_model.parent.mkdir(parents=True,exist_ok=True)
    a.out_metrics.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(result,a.out_model)
    report={'status':'V35_MULTISPEED_OFFLINE_TEACHER_IMITATION',
            'model_path':str(a.out_model),'speeds_mps':result['train_speed_targets_mps'],
            'metrics':result['metrics'],'train_groups':len(result['train_seed_groups']),
            'val_groups':len(result['val_seed_groups']),
            'limitations':result['limitations']}
    a.out_metrics.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for speed,maps in result['metrics']['val']['by_target_speed_and_map'].items():
        for name,metrics in maps.items():
            print(f'VAL speed={speed} map={name} PDD_MAE={metrics["original_pdd"]["steering_mae"]:.5f} '
                  f'V35_MAE={metrics["student_v35"]["steering_mae"]:.5f} improved={metrics["improved_mae"]}')
    print('Saved:',a.out_model.resolve(),a.out_metrics.resolve())
    print('OFFLINE imitation metrics are not actual autonomous driving performance.')


if __name__=='__main__':main()
