"""Behavior cloning of V3.8 traffic-following RULE teacher from executed actions."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from common_v38 import VERSION,STATUS,feature_names,validate_splits


def read_csv(path,features):
    rows=[]
    with Path(path).open(newline='',encoding='utf8') as f:
        reader=csv.DictReader(f)
        for item in reader:
            row=np.array([float(item[k]) for k in features],dtype=float)
            target=float(item['teacher_longitudinal'])
            if not np.isfinite(row).all() or not np.isfinite(target) or not -1<=target<=1:
                raise ValueError(f'Invalid sample in {path}')
            if abs(float(item['applied_longitudinal'])-target)>1e-8:
                raise ValueError('Teacher label != executed longitudinal')
            rows.append((row,target,item['map'],int(item['seed']),
                         float(item['base_speed_mps']),item['scenario']))
    if not rows:raise ValueError(f'No rows in {path}')
    return (np.stack([r[0] for r in rows]),np.array([r[1] for r in rows]),rows)


def summarize(y,pred):
    brakes=y<-.02
    return {'n':len(y),'mae':float(mean_absolute_error(y,pred)),
            'rmse':float(np.sqrt(mean_squared_error(y,pred))),
            'brake_n':int(brakes.sum()),
            'brake_mae':float(np.mean(np.abs(y[brakes]-pred[brakes]))) if brakes.any() else None,
            'brake_recall_at_0p02':float(np.mean(pred[brakes]<-.02)) if brakes.any() else None}


def train(data_dir,pdd_features,iterations=260):
    manifest=json.loads((Path(data_dir)/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!=STATUS:raise ValueError('Wrong dataset')
    splits=validate_splits(manifest['splits']);features=feature_names(pdd_features)
    if manifest['data_fields'][-len(features):]!=features:raise ValueError('Feature schema mismatch')
    train_x,train_y,train_rows=read_csv(Path(data_dir)/'train.csv',features)
    val_x,val_y,val_rows=read_csv(Path(data_dir)/'val.csv',features)
    for split,rows in [('train',train_rows),('val',val_rows)]:
        if set(r[3] for r in rows)-set(splits[split]):raise ValueError('Split leakage')
        if len(rows)!=manifest['accepted_rows'][split]:raise ValueError('Manifest count mismatch')
        if len(rows)<300:raise ValueError('Not enough samples')
        for scenario in manifest['scenarios']:
            if sum(r[5]==scenario for r in rows)<50:
                raise ValueError(f'Missing training coverage for {split}/{scenario}')
        if sum(r[0][-5]>0.5 for r in rows)<100:
            raise ValueError(f'No meaningful observed lead samples for {split}')
        if sum(r[1]<-.02 and r[0][-5]>.5 for r in rows)<20:
            raise ValueError(f'Insufficient lead-related brake samples for {split}')
    if set(r[3] for r in train_rows)&set(r[3] for r in val_rows):raise ValueError('Seed collision')
    # Balance scenario/speed; balance positive lead encounters vs no front.
    from collections import Counter
    counts=Counter((r[4],r[5],int(r[0][-5]>.5)) for r in train_rows)
    weight=np.array([len(train_rows)/(len(counts)*counts[(r[4],r[5],int(r[0][-5]>.5))])
                     for r in train_rows],dtype=float)
    weight=np.minimum(weight,8.);weight/=weight.mean()
    model=HistGradientBoostingRegressor(max_iter=iterations,learning_rate=.05,
          max_leaf_nodes=31,min_samples_leaf=25,l2_regularization=2.,random_state=42,
          early_stopping=False)
    model.fit(train_x,train_y,sample_weight=weight)
    metrics={}
    for name,x,y,rows in [('train',train_x,train_y,train_rows),('val',val_x,val_y,val_rows)]:
        pred=np.clip(model.predict(x),-1.,1.)
        mask=np.array([r[0][-5]>.5 for r in rows])
        metrics[name]={'all':summarize(y,pred),'front_present':summarize(y[mask],pred[mask]),
                       'front_absent':summarize(y[~mask],pred[~mask]) if (~mask).any() else None}
    return {'version':VERSION,'model':model,'pdd_features':list(pdd_features),
            'feature_order':features,'splits':splits,'speeds_mps':manifest['speeds_mps'],
            'scenarios':manifest['scenarios'],'teacher_label':'executed handcrafted gap-aware policy (NOT IDM)',
            'metrics':metrics,'limitations':['Controlled scripted lead, S maps only.',
               'Teacher is rule-based; learned response not a proof of physical safety.',
               'No random traffic, pedestrians, intersections, or lane-change training.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',required=True,type=Path);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--out-model',type=Path,default=Path('agent4_v38_traffic.joblib'))
    p.add_argument('--out-metrics',type=Path,default=Path('agent4_v38_training_metrics.json'))
    p.add_argument('--iterations',type=int,default=260)
    a=p.parse_args()
    if not 50<=a.iterations<=600:p.error('iterations outside 50..600')
    pdd=joblib.load(a.pdd)
    if pdd.get('label_mode')!='at_t' or len(pdd.get('features',[]))!=15:
        p.error('Wrong PDD')
    bundle=train(a.data_dir,pdd['features'],a.iterations)
    a.out_model.parent.mkdir(parents=True,exist_ok=True)
    a.out_metrics.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,a.out_model)
    a.out_metrics.write_text(json.dumps({'version':VERSION,'teacher':bundle['teacher_label'],
                       'splits':bundle['splits'],'metrics':bundle['metrics'],
                       'limitations':bundle['limitations']},indent=2,allow_nan=False),encoding='utf8')
    print('Saved',a.out_model,a.out_metrics,'validation',bundle['metrics']['val'])
if __name__=='__main__':main()
