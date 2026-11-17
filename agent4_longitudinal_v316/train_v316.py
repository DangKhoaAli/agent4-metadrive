"""Train a stand-alone signed gas/brake model from V3.16 TRAIN shadow IDM labels.

Uses V3.12B TRAIN verified IDM demonstrations as anchors. V3.16 VAL is never
used to fit model. Requires meaningful braking supervision in both splits.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import joblib,numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingRegressor
from common_v316 import (VERSION,dependencies,make_features,model_features,sha)


def read_collection(root,report,feat,splits):
    arr={sp:dict(x=[],y=[],cell=[],valid=0,brake=0,episodes=0) for sp in ('train','val')}
    config=report['config']; expected=(len(config['train_seeds'])+len(config['val_seeds']))*len(config['maps'])*len(config['speeds_mps'])
    if len(report['episodes'])!=expected:raise ValueError(f'Incomplete collection: {len(report["episodes"])}/{expected}')
    unique=set()
    for ep in report['episodes']:
        sp=ep['split'];m=ep['map'];seed=int(ep['seed']);v=float(ep['speed_mps'])
        if sp not in ('train','val') or seed not in splits[sp]:raise ValueError('Train/val seed leakage')
        if seed not in config[f'{sp}_seeds'] or m not in config['maps'] or v not in config['speeds_mps']:
            raise ValueError('Unexpected episode identification')
        key=(sp,m,seed,v)
        if key in unique:raise ValueError('Duplicate collection episode')
        unique.add(key)
        if ep['errors'] or ep['steps']!=ep['verified_learner_steps'] or ep['steps']<=0 or ep.get('teacher_driving_calls')!=0:
            raise ValueError(f'Learner action verification failed: {key}: {ep["errors"]}')
        p=(root/ep['csv']).resolve()
        if root not in p.parents or not p.is_file():raise ValueError('Invalid/missing episode CSV')
        accepted=0;brakes=0;total=0
        with p.open('r',encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            required=set(feat.FEATURES)|{'target_speed_mps','teacher_longitudinal','teacher_label_valid','split','map','seed','base_speed_mps','teacher_label_only'}
            if not required<=set(reader.fieldnames or []):raise ValueError(f'Missing CSV columns {required-set(reader.fieldnames or [])}')
            for row in reader:
                total+=1
                if (row['split'],row['map'],int(row['seed']),float(row['base_speed_mps']))!=key or row['teacher_label_only']!='1':
                    raise ValueError('Wrong episode metadata or non-shadow IDM label')
                if row['teacher_label_valid']!='1':continue
                y=float(row['teacher_longitudinal'])
                if not math.isfinite(y) or abs(y)>1.0001:raise ValueError('Invalid teacher label')
                arr[sp]['x'].append(make_features(row,feat));arr[sp]['y'].append(y)
                arr[sp]['cell'].append((m,v));accepted+=1;brakes+=int(y<-.02)
        if total!=ep['steps'] or accepted!=ep['valid_labels'] or brakes!=ep['teacher_brake_labels']:
            raise ValueError(f'Manifest/CSV mismatch {key}')
        arr[sp]['valid']+=accepted;arr[sp]['brake']+=brakes;arr[sp]['episodes']+=1
    if arr['train']['valid']<500 or arr['val']['valid']<100:raise ValueError('Not enough valid shadow labels')
    if arr['train']['brake']<30 or arr['val']['brake']<10:
        raise ValueError(f'Insufficient teacher BRAKE examples: train={arr["train"]["brake"]}, val={arr["val"]["brake"]}; do not claim learned braking')
    for sp in ('train','val'):
        arr[sp]['x']=np.asarray(arr[sp]['x'],dtype=np.float32)
        arr[sp]['y']=np.asarray(arr[sp]['y'],dtype=np.float32)
    return arr


def load_train_anchors(dataset,manifest,feat,cap=12000):
    root=Path(dataset).resolve()
    eps=[e for e in manifest['episodes'] if e['split']=='train' and e.get('clean_file')]
    if not eps:raise ValueError('No verified original TRAIN clean CSV anchors')
    xs=[];ys=[]; by_cell=Counter()
    per=max(1,cap//len(eps))
    for ep in eps:
        p=(root/ep['clean_file']).resolve()
        if root not in p.parents or not p.is_file():raise ValueError(f'Missing verified train clean CSV {p}')
        stride=max(1,math.ceil(int(ep['actual_trainable_steps'])/per))
        with p.open('r',encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):
                if i%stride:continue
                if row['split']!='train' or int(row['seed'])!=int(ep['seed']) or row['trainable_row']!='1' or row['current_action_verified']!='1' or row['label_valid']!='1':
                    raise ValueError(f'Unverified original IDM TRAIN label in {p}')
                y=float(row['applied_throttle_brake'])
                if not math.isfinite(y) or abs(y)>1.0001:raise ValueError('Invalid original anchor')
                xs.append(make_features(row,feat));ys.append(y)
                by_cell[(ep['map'],float(ep['speed_mps']))]+=1
    if not xs:raise ValueError('No original anchors')
    return np.asarray(xs,np.float32),np.asarray(ys,np.float32),by_cell


def measure(y,pred):
    pred=np.clip(np.asarray(pred,float),-1.,1.);y=np.asarray(y,float)
    neg=y<-.02;pos=y>.02
    return dict(n=int(y.size),mae=float(np.mean(abs(y-pred))),
        rmse=float(np.sqrt(np.mean((y-pred)**2))),
        brake_n=int(neg.sum()),
        brake_mae=float(np.mean(abs(y[neg]-pred[neg]))) if neg.any() else None,
        brake_recall=float((pred[neg]<-.02).mean()) if neg.any() else None,
        throttle_n=int(pos.sum()),
        throttle_mae=float(np.mean(abs(y[pos]-pred[pos]))) if pos.any() else None,
        predicted_brake_ratio=float((pred<-.02).mean()))


def train(args):
    c,feat,_=dependencies(args.legacy_root)
    orig,splits,orig_hash=c.original_manifest(args.dataset)
    baseline=joblib.load(args.baseline)
    c.check_baseline(baseline,orig_hash,feat)
    root=Path(args.collection).resolve();mp=root/'collection_manifest.json'
    doc=json.loads(mp.read_text(encoding='utf-8'))
    if doc.get('version')!=VERSION or doc['config']['original_manifest_sha256']!=orig_hash or doc['config']['baseline_sha256']!=sha(args.baseline):
        raise ValueError('Collection mismatch with original dataset/baseline model')
    print('Loading verified V3.16 TRAIN and VAL shadow labels...',flush=True)
    dat=read_collection(root,doc,feat,splits)
    print(f'Shadow labels: TRAIN {dat["train"]["valid"]:,} ({dat["train"]["brake"]:,} brake), VAL {dat["val"]["valid"]:,} ({dat["val"]["brake"]:,} brake)',flush=True)
    ax,ay,cell=load_train_anchors(args.dataset,orig,feat)
    # Use only TRAIN states for fitting. Weight braking samples more heavily.
    tx=dat['train']['x'];ty=dat['train']['y']
    x=np.concatenate((tx,ax),axis=0);y=np.concatenate((ty,ay),axis=0)
    w=np.concatenate((np.where(ty<-.02,3.0,1.0),np.full(len(ay),.4)))
    if args.iterations<50 or args.iterations>600:raise ValueError('Iterations outside [50,600]')
    est=HistGradientBoostingRegressor(max_iter=args.iterations,learning_rate=.06,
        max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1.0,
        random_state=42,early_stopping=False)
    print(f'Training learned signed gas/brake on {len(tx):,} shadow + {len(ax):,} IDM anchors...',flush=True)
    est.fit(x,y,sample_weight=w)
    metrics={sp:measure(dat[sp]['y'],est.predict(dat[sp]['x'])) for sp in ('train','val')}
    metrics['val_by_map']={}
    for m in doc['config']['maps']:
        idx=np.array([mm==m for mm,_ in dat['val']['cell']])
        if idx.any():metrics['val_by_map'][m]=measure(dat['val']['y'][idx],est.predict(dat['val']['x'][idx]))
    bundle=dict(version=VERSION,estimator=est,feature_names=list(model_features(feat)),
        original_manifest_sha256=orig_hash,baseline_sha256=sha(args.baseline),
        collection_manifest_sha256=sha(mp),teacher_fallback=False,learned_longitudinal=True,
        trained_on_split='train',model_selection_split='val',
        label_source='counterfactual shadow IDM + verified original train IDM anchors',
        steering_reference='V3.13 frozen; no teacher steering at inference',
        shadow_train_seeds=doc['config']['train_seeds'],shadow_val_seeds=doc['config']['val_seeds'])
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:raise FileExistsError('Output directory not empty; supply --overwrite only intentionally')
    out.mkdir(parents=True,exist_ok=True)
    model_path=out/'agent4_v316_longitudinal.joblib'
    joblib.dump(bundle,model_path,compress=3)
    c.atomic_json(out/'metrics_v316.json',dict(version=VERSION,
        train_shadow_rows=len(tx),train_anchor_rows=len(ax),val_shadow_rows=len(dat['val']['y']),
        train_brake_labels=dat['train']['brake'],val_brake_labels=dat['val']['brake'],
        metrics=metrics,original_manifest_sha256=orig_hash,
        note='Offline imitation is NOT closed-loop approval; no previous holdout seed was read.'))
    print('VAL:',metrics['val'],flush=True)
    print('Saved:',model_path.resolve(),flush=True)
    return bundle


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--collection',type=Path,default=Path('agent4_v316_data'))
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v316_model'))
    p.add_argument('--iterations',type=int,default=180)
    p.add_argument('--overwrite',action='store_true')
    return train(p.parse_args(argv))

if __name__=='__main__':main()
