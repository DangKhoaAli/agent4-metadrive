"""Train V3.16.2 longitudinal DAgger using V3.16 TRAIN + on-policy TRAIN + original TRAIN anchors.

Validation is scored but NEVER included in estimator.fit. No held-out seed <=201 is reused as final.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from collections import Counter
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from common_v3162 import (VERSION,dependencies,make_features,model_features,sha,validate_model)
from train_v316 import read_collection,load_train_anchors,measure


def read_dagger(root,doc,feat,splits):
    cfg=doc['config']; seeds=set(map(int,cfg['train_seeds']))
    if not seeds or not seeds<=set(splits['train']):
        raise ValueError('DAgger TRAIN seeds are not original TRAIN seeds')
    if cfg.get('behavior_policy')!='frozen V3.16 gas/brake + V3.13 steering':
        raise ValueError('DAgger was not collected by the frozen learner')
    wanted={(m,int(s),float(v)) for m in cfg['maps'] for s in seeds for v in cfg['speeds_mps']}
    if len(doc['episodes'])!=len(wanted):raise ValueError(f'Incomplete DAgger collection {len(doc["episodes"])} / {len(wanted)}')
    seen=set();x=[];y=[];cells=[];brakes=0;status=Counter()
    root=Path(root).resolve()
    for ep in doc['episodes']:
        key=(ep['map'],int(ep['seed']),float(ep['speed_mps']))
        if (ep.get('split')!='train' or key not in wanted or key in seen or
            ep.get('teacher_driving_calls')!=0 or ep.get('errors') or
            int(ep['steps'])!=int(ep['verified_learner_steps']) or ep['steps']<=0):
            raise ValueError(f'Unverified or duplicate DAgger episode: {key}')
        seen.add(key);status[ep['status']]+=1
        p=(root/ep['csv']).resolve()
        if root not in p.parents or not p.is_file():raise ValueError('Unsafe or missing DAgger CSV')
        n=ok_n=brake_n=0
        with p.open('r',newline='',encoding='utf-8-sig') as f:
            rdr=csv.DictReader(f)
            required=set(feat.FEATURES)|{'target_speed_mps','split','map','seed','base_speed_mps',
                'teacher_longitudinal','teacher_label_valid','teacher_label_only','applied_longitudinal'}
            if not required<=set(rdr.fieldnames or []):raise ValueError('Missing DAgger columns')
            for row in rdr:
                n+=1
                if (row['split']!='train' or row['map']!=key[0] or int(row['seed'])!=key[1] or
                    float(row['base_speed_mps'])!=key[2] or row['teacher_label_only']!='1'):
                    raise ValueError('DAgger data metadata mismatch')
                if not -1.0005<=float(row['applied_longitudinal'])<=1.0005:
                    raise ValueError('Invalid applied learned action')
                if row['teacher_label_valid']!='1':continue
                label=float(row['teacher_longitudinal'])
                if not math.isfinite(label) or abs(label)>1.0001:raise ValueError('Invalid DAgger label')
                x.append(make_features(row,feat));y.append(label);cells.append((key[0],key[2]));ok_n+=1
                brake_n+=int(label<-.02)
        if n!=ep['steps'] or ok_n!=ep['valid_labels'] or brake_n!=ep['teacher_brake_labels']:
            raise ValueError('DAgger CSV differs from manifest counts')
        brakes+=brake_n
    if set(seen)!=wanted:raise ValueError('Missing DAgger cells')
    if len(x)<500 or brakes<30:raise ValueError('Insufficient valid on-policy training labels / brakes')
    return dict(x=np.asarray(x,np.float32),y=np.asarray(y,np.float32),
                cells=cells,brakes=brakes,episodes=len(seen),status=dict(status))


def train(args):
    c,feat,_=dependencies(args.legacy_root)
    original,splits,original_hash=c.original_manifest(args.dataset)
    baseline=joblib.load(args.baseline)
    c.check_baseline(baseline,original_hash,feat)
    baseline_sha=sha(args.baseline);actor_sha=sha(args.actor)
    actor=joblib.load(args.actor)
    validate_model(actor,original_hash,baseline_sha,feat)
    oldroot=Path(args.v316_collection).resolve();oldmanifest=oldroot/'collection_manifest.json'
    olddoc=json.loads(oldmanifest.read_text(encoding='utf-8'))
    if (olddoc['version']!='AGENT4_V316_SHADOW_IDM_LONGITUDINAL' or
        olddoc['config']['original_manifest_sha256']!=original_hash or
        olddoc['config']['baseline_sha256']!=baseline_sha):
        raise ValueError('Original V3.16 collection provenance mismatch')
    old=read_collection(oldroot,olddoc,feat,splits)
    dagroot=Path(args.dagger_collection).resolve();dagmanifest=dagroot/'collection_manifest.json'
    dagdoc=json.loads(dagmanifest.read_text(encoding='utf-8'))
    if (dagdoc.get('version')!=VERSION or dagdoc['config']['actor_sha256']!=actor_sha or
        dagdoc['config']['baseline_sha256']!=baseline_sha or
        dagdoc['config']['original_manifest_sha256']!=original_hash):
        raise ValueError('DAgger dataset provenance mismatch')
    dag=read_dagger(dagroot,dagdoc,feat,splits)
    ax,ay,_=load_train_anchors(args.dataset,original,feat)
    x=np.concatenate([old['train']['x'],dag['x'],ax],axis=0)
    y=np.concatenate([old['train']['y'],dag['y'],ay],axis=0)
    # DAgger visited states matter more; reduce brake overweight vs V3.16 (3x).
    w=np.concatenate([np.where(old['train']['y']<-.02,1.25,1.0),
                      np.where(dag['y']<-.02,1.875,1.5),np.full(len(ay),0.30)])
    if args.iterations<50 or args.iterations>600:raise ValueError('iterations must be [50,600]')
    estimator=HistGradientBoostingRegressor(max_iter=args.iterations,learning_rate=.05,
        max_leaf_nodes=25,min_samples_leaf=50,l2_regularization=2.0,
        random_state=42,early_stopping=False)
    print(f'Training {len(old["train"]["y"]):,} original TRAIN shadow + {len(dag["y"]):,} on-policy TRAIN + {len(ay):,} verified TRAIN anchors',flush=True)
    estimator.fit(x,y,sample_weight=w)
    metrics={'original_shadow_train':measure(old['train']['y'],estimator.predict(old['train']['x'])),
             'on_policy_train':measure(dag['y'],estimator.predict(dag['x'])),
             'validation_not_used_for_fit':measure(old['val']['y'],estimator.predict(old['val']['x']))}
    metrics['val_by_map']={}
    for m in olddoc['config']['maps']:
        mask=np.array([x[0]==m for x in old['val']['cell']])
        if mask.any():metrics['val_by_map'][m]=measure(old['val']['y'][mask],estimator.predict(old['val']['x'][mask]))
    bundle=dict(version=VERSION,estimator=estimator,feature_names=list(model_features(feat)),
         original_manifest_sha256=original_hash,baseline_sha256=baseline_sha,
         actor_sha256=actor_sha,original_collection_sha256=sha(oldmanifest),
         dagger_collection_sha256=sha(dagmanifest),teacher_fallback=False,
         learned_longitudinal=True,trained_on_split='train',
         label_source='V3.16 TRAIN shadow IDM + V3.16 on-policy TRAIN shadow IDM + verified TRAIN IDM anchors',
         steering_reference='Frozen V3.13; no teacher control in evaluation',
         dagger_train_seeds=sorted(set(map(int,dagdoc['config']['train_seeds']))),
         validation_seeds=olddoc['config']['val_seeds'])
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('New model directory required; no overwrite of past models')
    out.mkdir(parents=True,exist_ok=True)
    path=out/'agent4_v3162_longitudinal.joblib'
    joblib.dump(bundle,path,compress=3)
    c.atomic_json(out/'metrics_v3162.json',dict(version=VERSION,
         old_train_rows=len(old['train']['y']),dagger_train_rows=len(dag['y']),
         anchor_train_rows=len(ay),val_rows=len(old['val']['y']),
         dagger_train_brakes=dag['brakes'],dagger_episodes=dag['episodes'],
         dagger_status=dag['status'],metrics=metrics,baseline_sha256=baseline_sha,
         actor_sha256=actor_sha,original_manifest_sha256=original_hash,
         note='Training used only TRAIN rows. Offline metrics do not approve closed-loop use.'))
    print('Untouched VAL:',metrics['validation_not_used_for_fit'],flush=True)
    print('Saved:',path.resolve(),flush=True)
    return bundle


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--v316-collection',type=Path,default=Path('agent4_v316_data'))
    p.add_argument('--dagger-collection',type=Path,default=Path('agent4_v3162_dagger_data'))
    p.add_argument('--iterations',type=int,default=180)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v3162_model'))
    return train(p.parse_args(argv))

if __name__=='__main__':main()
