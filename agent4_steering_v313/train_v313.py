"""Train V3.13 steering ONLY from complete V3.12B clean CSVs.

Run from the dataset directory:
  python agent4_steering_v313/train_v313.py --dataset .\\agent4_v312b_idm_dataset
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

from features_v313 import FEATURES,LABEL,VERSION,DATA_VERSION,row_features

SPLITS=('train','val','test')


def manifest_load(dataset):
    path=Path(dataset)/'manifest.json'
    if not path.is_file():raise FileNotFoundError(f'{path} missing. Need full V3.12B dataset folder, not manifest alone')
    doc=json.loads(path.read_text(encoding='utf-8'))
    if doc.get('version')!=DATA_VERSION:raise ValueError(f'Not a V3.12B dataset: {doc.get("version")}')
    splits=doc['config']['split_seeds']
    if set(splits)!=set(SPLITS):raise ValueError('Manifest must contain train, val, test')
    all_seeds=[(split,int(x)) for split,arr in splits.items() for x in arr]
    if len(set(seed for _,seed in all_seeds))!=len(all_seeds):
        raise ValueError('Seed leakage: train/val/test overlap')
    if any(not splits[sp] for sp in SPLITS):raise ValueError('All three splits must contain seeds')
    expected=len(doc['config']['maps'])*len(doc['config']['speeds_mps'])*len(all_seeds)
    if len(doc.get('episodes',[]))!=expected:
        raise ValueError(f'Dataset incomplete: {len(doc.get("episodes",[]))}/{expected} episodes. Do not train yet')
    identities=set()
    for ep in doc['episodes']:
        ident=(ep['split'],ep['map'],int(ep['seed']),float(ep['speed_mps']))
        if ident in identities:raise ValueError(f'Duplicate episode {ident}')
        identities.add(ident)
        if ident[0] not in splits or ident[2] not in splits[ident[0]]:
            raise ValueError(f'Wrong split assignment in {ident}')
        if ident[1] not in doc['config']['maps'] or ident[3] not in doc['config']['speeds_mps']:
            raise ValueError(f'Unexpected episode map/speed: {ident}')
        if not ep.get('quality_pass') or ep.get('status')!='arrive_dest':
            raise ValueError(f'Episode {ident} not approved; refusal to train')
        if not ep.get('recorded_actions_aligned') or ep.get('errors'):
            raise ValueError(f'Invalid action trace in {ident}')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return doc,sha


def load_samples(dataset,doc):
    dataset=Path(dataset).resolve()
    groups={s:dict(X=[],y=[],map=[],speed=[],circular=[],episode=[],weight=[]) for s in SPLITS}
    for ep_id,ep in enumerate(doc['episodes']):
        sp=ep['split'];rel=ep.get('clean_file')
        if not rel or not str(rel).replace('\\','/').startswith('clean/'):
            raise ValueError(f'Invalid clean path in episode #{ep_id}: {rel!r}')
        path=(dataset/rel).resolve()
        if dataset not in path.parents:raise ValueError(f'CSV escaped dataset directory: {path}')
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; upload/copy the COMPLETE clean CSV directory')
        count=0
        with path.open('r',encoding='utf-8-sig',newline='') as fh:
            reader=csv.DictReader(fh)
            needed=set(FEATURES)|{LABEL,'trainable_row','current_action_verified','label_valid',
                 'map','seed','split','target_speed_mps'}
            if not needed.issubset(set(reader.fieldnames or [])):
                raise ValueError(f'CSV schema mismatch at {path}: {sorted(needed-set(reader.fieldnames or []))}')
            for row in reader:
                if row['trainable_row']!='1' or row['current_action_verified']!='1' or row['label_valid']!='1':
                    raise ValueError(f'Invalid supervision in clean CSV {path} row {count+2}')
                if row['split']!=sp or row['map']!=ep['map'] or int(row['seed'])!=int(ep['seed']) or \
                        abs(float(row['target_speed_mps'])-float(ep['speed_mps']))>1e-8:
                    raise ValueError(f'Episode metadata mismatch in clean CSV {path}')
                x=row_features(row)
                try:y=float(row[LABEL])
                except ValueError:raise ValueError(f'Missing steering label in {path}') from None
                if not math.isfinite(y) or abs(y)>1.00001:raise ValueError(f'Invalid steering label {y} in {path}')
                g=groups[sp]
                g['X'].append(x);g['y'].append(y)
                g['map'].append(ep['map']);g['speed'].append(float(ep['speed_mps']))
                g['circular'].append(int(float(row['is_circular_lane'])))
                g['episode'].append(ep_id)
                count+=1
        if count!=int(ep['actual_trainable_steps']):
            raise ValueError(f'CSV rows {count} != manifest {ep["actual_trainable_steps"]}: {path}')
    for sp,g in groups.items():
        g['X']=np.asarray(g['X'],dtype=np.float32).reshape((-1,len(FEATURES)))
        g['y']=np.asarray(g['y'],dtype=np.float32)
        for key in ('map','speed','circular','episode'):
            g[key]=np.asarray(g[key])
        if not len(g['y']):raise ValueError(f'Empty split: {sp}')
        _,inverse,counts=np.unique(g['episode'],return_inverse=True,return_counts=True)
        w=1.0/counts[inverse]
        g['weight']=(w/w.mean()).astype(np.float32)
        expected=int(doc['totals']['by_split'][sp]['clean_steps'])
        if len(g['y'])!=expected:raise ValueError(f'Manifest {sp} count {expected}, loaded {len(g["y"])}')
    return groups


def metrics(y,pred):
    if not len(y):return None
    return dict(n=int(len(y)),mae=float(mean_absolute_error(y,pred)),
                rmse=float(np.sqrt(mean_squared_error(y,pred))),
                p95_abs_error=float(np.percentile(np.abs(y-pred),95)))


def evaluate(g,pred):
    pred=np.clip(np.asarray(pred,dtype=np.float64),-1.,1.)
    y=g['y']
    res={'overall':metrics(y,pred),'by_map':{},'by_map_speed':{},'by_lane_geometry':{}}
    for m in sorted(set(g['map'])):
        select=g['map']==m
        res['by_map'][m]=metrics(y[select],pred[select])
        for v in sorted(set(g['speed'])):
            keep=select&(g['speed']==v)
            if np.any(keep):res['by_map_speed'][f'{m}@{v:g}mps']=metrics(y[keep],pred[keep])
    for state,tag in [(0,'non_circular'),(1,'circular')]:
        keep=g['circular']==state
        if np.any(keep):res['by_lane_geometry'][tag]=metrics(y[keep],pred[keep])
    return res


def selection_score(g,pred):
    """Validation only; independent map/speed buckets; C/X are emphasized."""
    pred=np.clip(np.asarray(pred),-1.,1.)
    num=denom=0.
    for m in set(g['map']):
        for v in set(g['speed']):
            keep=(g['map']==m)&(g['speed']==v)
            if not np.any(keep):continue
            weight=1. if m=='S' else 2.
            num+=weight*float(np.mean(np.abs(g['y'][keep]-pred[keep])))
            denom+=weight
    return num/denom


def candidates(names):
    all_models={
        'ridge':lambda:make_pipeline(StandardScaler(),Ridge(alpha=5.)),
        'hgb':lambda:HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=31,
                    min_samples_leaf=40,learning_rate=.065,l2_regularization=0.3,
                    early_stopping=False,random_state=42),
        'extra':lambda:ExtraTreesRegressor(n_estimators=96,min_samples_leaf=4,
                     max_features=.9,n_jobs=-1,random_state=42),
    }
    out=[]
    for name in names.split(','):
        key=name.strip().lower()
        if key not in all_models:raise ValueError(f'Unknown model {key!r}; valid {tuple(all_models)}')
        if key in [x[0] for x in out]:raise ValueError(f'Duplicate candidate: {key}')
        out.append((key,all_models[key]()))
    if not out:raise ValueError('Empty model list')
    return out


def run(args):
    import sklearn
    data_root=Path(args.dataset)
    doc,manifest_sha=manifest_load(data_root)
    print('Dataset manifest verified; loading all clean CSVs...',flush=True)
    splits=load_samples(data_root,doc)
    print('Loaded:',', '.join(f'{k}={len(v["y"]):,}' for k,v in splits.items()),flush=True)
    val=splits['val'];train=splits['train'];test=splits['test']
    results=[];winner=None;winner_score=math.inf
    zero_report=evaluate(val,np.zeros_like(val['y']))
    for name,estimator in candidates(args.models):
        print(f'Training {name} ...',flush=True)
        if name=='ridge':
            estimator.fit(train['X'],train['y'],ridge__sample_weight=train['weight'])
        else:
            estimator.fit(train['X'],train['y'],sample_weight=train['weight'])
        val_pred=estimator.predict(val['X'])
        score=selection_score(val,val_pred)
        print(f'  VAL balanced MAE={score:.6f}, MAE={mean_absolute_error(val["y"],np.clip(val_pred,-1,1)):.6f}',flush=True)
        results.append({'model':name,'val_balanced_score':score,
                        'val_metrics':evaluate(val,val_pred)})
        if score<winner_score:
            winner_score=score;winner=(name,estimator)
    name,estimator=winner
    # Model chosen with validation ONLY. Test is read once here, after selection.
    test_pred=estimator.predict(test['X'])
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{out} already contains files; choose new --out-dir or --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    bundle=dict(version=VERSION,estimator=estimator,features=list(FEATURES),
                label=LABEL,trained_on_manifest_sha256=manifest_sha,
                trained_on_split='train',hyperparameter_selection_split='val',
                learner=name,learned_longitudinal=False,
                teacher_fallback=False,geometry_source='privileged MetaDrive navigation',
                dependency_versions={'sklearn':sklearn.__version__,'numpy':np.__version__})
    joblib.dump(bundle,out/'agent4_v313_steering.joblib',compress=3)
    report=dict(version=VERSION,manifest_sha256=manifest_sha,
                dataset_path=str(data_root.resolve()),
                n_samples={sp:len(g['y']) for sp,g in splits.items()},
                features=list(FEATURES),label=LABEL,selected_model=name,
                selection_metric='weighted average of validation MAE per (map,speed), C/X weight=2 S weight=1',
                selection_score=winner_score,val_zero_steering_baseline=zero_report,
                candidate_validation=results,
                selected_model_test=evaluate(test,test_pred),
                selected_model_test_zero_baseline=evaluate(test,np.zeros_like(test['y'])),
                warnings=['Offline imitation error does not imply closed-loop arrival.',
                          'Test split was not used for model selection or tuning.',
                          'Steering learned; longitudinal remains rule-based in V3.13 evaluator.',
                          'Privileged simulator lane geometry, zero traffic, no real-world safety claims.'])
    (out/'metrics_v313.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Selected:',name,'VAL balanced MAE=',round(winner_score,6),flush=True)
    print('TEST MAE=',round(report['selected_model_test']['overall']['mae'],6),flush=True)
    print('Saved:',(out/'agent4_v313_steering.joblib').resolve(),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True,help='Complete V3.12B dataset, including clean/ and manifest.json')
    p.add_argument('--models',default='ridge,hgb,extra',help='Candidate list: ridge,hgb,extra')
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v313_model'))
    p.add_argument('--overwrite',action='store_true')
    return run(p.parse_args(argv))

if __name__=='__main__':main()
