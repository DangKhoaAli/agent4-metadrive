"""Train model to imitate ACTUALLY EXECUTED V3.5 governor, not IDM.

Frozen PDD+V3.5 steering drove teacher trajectories. Governor supplies labels.
Standalone model uses only observations at inference; no governor fallback.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_longitudinal_v36'))
from train_longitudinal_v36 import read_rows,evaluate
from common_v37 import VERSION,DATA_STATUS,names,validate_new_splits


def fit(data_dir,pdd_features,max_iter=300):
    root=Path(data_dir)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('status')!=DATA_STATUS:raise ValueError('V3.7 governor-teacher dataset required; IDM is a different teacher')
    splits=validate_new_splits(manifest['splits'])
    names_expected=names(pdd_features)
    if manifest['data_fields'][-len(names_expected):]!=names_expected:
        raise ValueError('Feature schema/order mismatch')
    train=read_rows(root/'train.csv',names_expected)
    val=read_rows(root/'val.csv',names_expected)
    for split,d in [('train',train),('val',val)]:
        x,y,maps,seeds,speeds,clipped=d
        if set(seeds)-set(splits[split]):raise ValueError('Seed leakage')
        if set(maps)!=set(manifest['maps']) or set(speeds)!=set(manifest['speeds_mps']):
            raise ValueError(f'Missing map/speed cells for {split}')
        if len(y)!=manifest['rows_by_split'][split]:raise ValueError('Manifest/CSV count mismatch')
        for m in manifest['maps']:
            for speed in manifest['speeds_mps']:
                if sum(mm==m and ss==speed for mm,ss in zip(maps,speeds))<40:
                    raise ValueError(f'Insufficient samples {split} map={m} speed={speed}')
    if set(train[3])&set(val[3]):raise ValueError('Train/val leakage')
    if sum(train[1]<-.02)<30 or sum(val[1]<-.02)<10:
        raise ValueError('Insufficient brake supervision')
    groups=list(zip(train[2],train[4]));count=Counter(groups)
    # Equal weighting per map/speed cell; emphasize small corrective actions at cruise.
    cruise=np.abs(train[1])<.08
    weights=np.asarray([len(groups)/(len(count)*count[g]) for g in groups],dtype=float)
    weights*=np.where(cruise,2.0,1.0)
    weights/=np.mean(weights)
    model=HistGradientBoostingRegressor(loss='squared_error',max_iter=max_iter,
        max_leaf_nodes=31,min_samples_leaf=30,learning_rate=.05,
        l2_regularization=1.0,early_stopping=False,random_state=42)
    model.fit(train[0],train[1],sample_weight=weights)
    metrics={}
    for label,data in [('train',train),('val',val)]:
        x,y,maps,seeds,speeds,clip=data
        pred=np.clip(model.predict(x),-1,1)
        by_cell={}
        for speed in sorted(set(speeds)):
            by_cell[f'{speed:g}']={}
            for m in sorted(set(maps)):
                mask=np.asarray([mm==m and ss==speed for mm,ss in zip(maps,speeds)])
                if mask.any():by_cell[f'{speed:g}'][m]=evaluate(y[mask],pred[mask])
        near_target=np.abs(y)<.02
        metrics[label]={'overall':evaluate(y,pred),'by_speed_and_map':by_cell,
             'cruise_action_mae':float(np.mean(np.abs(pred[near_target]-y[near_target]))) if near_target.any() else None,
             'clip_fraction':float(np.mean(clip))}
    bundle={'version':VERSION,'model':model,'feature_order':names_expected,
            'pdd_features':list(pdd_features),'maps':manifest['maps'],
            'speed_targets':manifest['speeds_mps'],'splits':splits,
            'teacher_label':'executed V3.5 rule-based governor longitudinal (NOT IDM)',
            'data_source':'on-policy steering V3.5 + governor longitudinal',
            'metrics':metrics,
            'limitations':['The teacher is a rule-based governor, not IDM.',
                           'Only steering V3.5 and speed governor drove recorded episodes.',
                           'Desired-speed schedule/planner remains rule-based.',
                           'No traffic; privileged simulator geometry.',
                           'Offline validation cannot establish closed-loop performance.']}
    return bundle


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdd',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('agent4_v37_governor_data'))
    p.add_argument('--out-model',type=Path,default=Path('agent4_v37_longitudinal.joblib'))
    p.add_argument('--out-metrics',type=Path,default=Path('agent4_v37_training_metrics.json'))
    p.add_argument('--max-iter',type=int,default=300)
    a=p.parse_args(argv)
    if not 50<=a.max_iter<=1000:p.error('max-iter must be 50..1000')
    pdd=joblib.load(a.pdd)
    if pdd.get('label_mode')!='at_t' or len(pdd.get('features',[]))!=15:
        raise ValueError('Expected PDD V2 at_t 15 features')
    b=fit(a.data_dir,pdd['features'],a.max_iter)
    a.out_model.parent.mkdir(parents=True,exist_ok=True)
    a.out_metrics.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(b,a.out_model)
    report={'status':'V37_GOVERNOR_ON_POLICY_OFFLINE_IMITATION',
           'train_splits':b['splits'],'teacher_label':b['teacher_label'],
           'metrics':b['metrics'],'limitations':b['limitations']}
    a.out_metrics.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('TRAIN',b['metrics']['train']['overall'])
    print('VAL  ',b['metrics']['val']['overall'])
    print('Saved model',a.out_model.resolve())

if __name__=='__main__':main()
