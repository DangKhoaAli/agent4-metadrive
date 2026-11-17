"""Train a standalone IDM longitudinal imitation model for use with V3.5 steering.

Requires actual V3.6 native IDM labels, NOT V3.5's governed_longitudinal.
Train/val grouped by seed. Equal weighting per map/base-speed cell. The output
is the signed MetaDrive longitudinal action: positive gas, negative brake.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from common_v36 import VERSION,names,check_splits,finite


def read_rows(path,features):
    with Path(path).open('r',encoding='utf-8',newline='') as fp:
        r=csv.DictReader(fp)
        mandatory=set(features)|{'teacher_longitudinal','applied_longitudinal','map','seed','base_speed_mps','step','longitudinal_clipped'}
        if not r.fieldnames or not mandatory<=set(r.fieldnames):
            raise ValueError(f'Missing required columns in {path}: {mandatory-set(r.fieldnames or [])}')
        x=[];y=[];seeds=[];maps=[];speeds=[];clipped=[]
        seen=set()
        for row in r:
            key=(row['map'],int(row['seed']),float(row['base_speed_mps']),int(row['step']))
            if key in seen:raise ValueError('Duplicate simulation sample: '+str(key))
            seen.add(key)
            feat=[finite(row[k],k) for k in features]
            target=finite(row['teacher_longitudinal'],'teacher longitudinal')
            applied=finite(row['applied_longitudinal'],'applied longitudinal')
            if abs(target-applied)>1e-8:
                raise ValueError('Teacher label differs from executed longitudinal action')
            if not -1.0001<=target<=1.0001:raise ValueError('Invalid longitudinal label')
            x.append(feat);y.append(target);maps.append(row['map']);seeds.append(int(row['seed']))
            speeds.append(float(row['base_speed_mps']));clipped.append(int(row['longitudinal_clipped']))
    if not x:raise ValueError(f'No data in {path}')
    return np.asarray(x,float),np.asarray(y,float),maps,seeds,speeds,np.asarray(clipped)


def validate_dataset(manifest,tr,va):
    if manifest.get('status')!='V36_NATIVE_IDM_LONGITUDINAL_TEACHER_DATA':
        raise ValueError('Manifest is not from V3.6 native IDM collection')
    splits=manifest['splits']
    check_splits(splits['train'],splits['val'],splits['eval'])
    expected_maps=set(manifest['maps']); expected_speeds=set(manifest['speeds_mps'])
    if set(tr[3]) - set(splits['train']) or set(va[3])-set(splits['val']):
        raise ValueError('Seed not in declared train/val split')
    if set(tr[2]) != expected_maps or set(va[2])!=expected_maps:
        raise ValueError('Map coverage mismatch')
    if set(tr[4])!=expected_speeds or set(va[4])!=expected_speeds:
        raise ValueError('Speed coverage mismatch')
    if len(tr[1])!=manifest['rows_by_split']['train'] or len(va[1])!=manifest['rows_by_split']['val']:
        raise ValueError('CSV/manifest row count mismatch')
    if set(tr[3]) & set(va[3]):raise ValueError('Train/val seed leakage')
    for label,arr in [('train',tr),('val',va)]:
        for m in expected_maps:
            for speed in expected_speeds:
                if sum(mm==m and ss==speed for mm,ss in zip(arr[2],arr[4]))<40:
                    raise ValueError(f'Insufficient {label} rows in map={m} speed={speed}')
    # Not enough braking supervision -> refuses to silently claim learned braking.
    if sum(tr[1]<-.02)<30 or sum(va[1]<-.02)<10:
        raise ValueError('Too few brake labels in train/val; inspect teacher speed schedule')


def cell_weights(maps,speeds):
    groups=list(zip(maps,speeds));counts=Counter(groups)
    n=len(groups)
    return np.asarray([n/(len(counts)*counts[g]) for g in groups],float)


def evaluate(y,pred):
    pred=np.clip(np.asarray(pred,float),-1.,1.)
    y=np.asarray(y,float)
    neg=y<-.02;pos=y>.02
    return {'n':int(y.size),
            'mae':float(mean_absolute_error(y,pred)),
            'rmse':float(np.sqrt(mean_squared_error(y,pred))),
            'brake_n':int(neg.sum()),'brake_mae':float(np.mean(abs(y[neg]-pred[neg]))) if neg.any() else None,
            'brake_recall_at_0p02':float((pred[neg]<-.02).mean()) if neg.any() else None,
            'throttle_n':int(pos.sum()),'throttle_mae':float(np.mean(abs(y[pos]-pred[pos]))) if pos.any() else None,
            'output_brake_fraction':float((pred<-.02).mean()),
            'teacher_brake_fraction':float(neg.mean())}


def fit(data_dir,pdd_features,max_iter=300):
    root=Path(data_dir)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    features=names(pdd_features)
    if manifest.get('data_fields',[])[-len(features):]!=features:
        raise ValueError('Manifest model feature order mismatch')
    tr=read_rows(root/'train.csv',features)
    va=read_rows(root/'val.csv',features)
    validate_dataset(manifest,tr,va)
    model=HistGradientBoostingRegressor(loss='squared_error',max_iter=max_iter,
        max_leaf_nodes=31,min_samples_leaf=40,learning_rate=.055,
        l2_regularization=1.5,random_state=42,early_stopping=False)
    model.fit(tr[0],tr[1],sample_weight=cell_weights(tr[2],tr[4]))
    metrics={}
    for split,arr in [('train',tr),('val',va)]:
        x,y,maps,seeds,speeds,clipped=arr
        pred=model.predict(x)
        by_speed_map={}
        for speed in sorted(set(speeds)):
            by_speed_map[str(speed)]={}
            for m in sorted(set(maps)):
                mask=np.array([s==speed and mm==m for s,mm in zip(speeds,maps)])
                if mask.any(): by_speed_map[str(speed)][m]=evaluate(y[mask],pred[mask])
        metrics[split]={'overall':evaluate(y,pred),
                        'by_speed_and_map':by_speed_map,
                        'clip_fraction':float(np.mean(clipped))}
    bundle={'version':VERSION,'model':model,'feature_order':features,
            'pdd_features':list(pdd_features),'splits':manifest['splits'],
            'maps':manifest['maps'],'speed_targets':manifest['speeds_mps'],
            'teacher_label':'executed clipped native IDM longitudinal action',
            'training_speed_schedule_enabled':manifest['speed_schedule_enabled'],
            'metrics':metrics,'limitations':['No traffic during collection or evaluation.',
                'Target speed is supplied by the route speed planner, not learned.',
                'IDM steering drives teacher collection; Student V3.5 steering drives evaluation.',
                'Train and validation use only successful IDM episodes.',
                'Offline loss alone does not prove the controller is stable.']}
    return bundle


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('agent4_v36_idm_data'))
    p.add_argument('--out-model',type=Path,default=Path('agent4_v36_longitudinal.joblib'))
    p.add_argument('--out-metrics',type=Path,default=Path('agent4_v36_training_metrics.json'))
    p.add_argument('--max-iter',type=int,default=300)
    a=p.parse_args(argv)
    if not 50<=a.max_iter<=1000:p.error('--max-iter 50..1000')
    frozen=joblib.load(a.pdd)
    if frozen.get('label_mode')!='at_t' or len(frozen.get('features',[]))!=15:
        raise ValueError('Expected frozen PDD at_t bundle (15 features)')
    bundle=fit(a.data_dir,frozen['features'],a.max_iter)
    a.out_model.parent.mkdir(parents=True,exist_ok=True)
    a.out_metrics.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,a.out_model)
    report={'status':'V36_LONGITUDINAL_OFFLINE_IMITATION',
            'model_path':str(a.out_model),'train_splits':bundle['splits'],
            'teacher_label':bundle['teacher_label'],
            'training_speed_schedule_enabled':bundle['training_speed_schedule_enabled'],
            'metrics':bundle['metrics'],'limitations':bundle['limitations']}
    a.out_metrics.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('TRAIN:',bundle['metrics']['train']['overall'])
    print('VAL:',bundle['metrics']['val']['overall'])
    print('Saved:',a.out_model.resolve(),a.out_metrics.resolve())
    print('NOTE: These are OFFLINE metrics; run V3.6 closed-loop comparison.')

if __name__=='__main__':main()
