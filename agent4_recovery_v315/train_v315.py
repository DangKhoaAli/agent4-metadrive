"""Train a gated residual steering model from on-policy TRAIN states only.

V3.13 stays frozen. V3.12B training demonstrations provide zero-residual anchors.
Validation is reported only; test split is neither read nor used for selection.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from common_v315 import (VERSION, atomic_json, check_baseline, digest,
                         legacy_modules, original_manifest, predict, recovery_gate)


def load_train_rows(collection_dir, manifest, features, splits):
    directory = Path(collection_dir).resolve()
    X, residual, weight, stats = [], [], [], {'considered':0,'selected':0,'episode_count':0}
    for ep in manifest['episodes']:
        if int(ep['seed']) not in splits['train']:
            raise ValueError('Training data contains a non-train seed')
        if ep['errors'] or ep['steps'] != ep['learner_steps_verified']:
            raise ValueError(f'Bad learner action alignment: {ep["map"]}/{ep["seed"]}')
        path = (directory / ep['csv']).resolve()
        if directory not in path.parents or not path.is_file():
            raise ValueError('Missing or outside-collection CSV: ' + str(path))
        stats['episode_count'] += 1
        with path.open('r',encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                stats['considered'] += 1
                if int(row['seed']) != ep['seed'] or row['map'] != ep['map'] or float(row['target_speed_mps']) != ep['speed_mps']:
                    raise ValueError('CSV episode metadata mismatch')
                if row['label_valid'] != '1': continue
                vec = features.row_features(row)
                label = float(row['teacher_steer'])
                base = float(row['base_steer'])
                if not math.isfinite(label) or not math.isfinite(base) or abs(label)>1.0001:
                    raise ValueError('Untrusted expert label')
                gate = recovery_gate(row)
                # Learn recovery where it matters, preserve demonstration manifold via anchors.
                X.append(vec); residual.append(label-base)
                weight.append(0.6 + 3.0*gate + float(int(float(row['is_circular_lane']))))
                stats['selected'] += 1
    if stats['selected'] < 500:
        raise ValueError('Too few valid on-policy expert labels; collect more TRAIN episodes')
    return X,residual,weight,stats


def load_anchors(dataset, dataset_manifest, baseline, features, max_total=14000):
    """Read only TRAIN clean CSVs, evenly sampling rows within each episode."""
    root = Path(dataset).resolve(); X=[]; y=[]; w=[]
    train_eps = [ep for ep in dataset_manifest['episodes'] if ep['split']=='train']
    if not train_eps: raise ValueError('No original train demonstrations')
    per_ep = max(1, max_total // len(train_eps))
    for ep in train_eps:
        path = (root / ep['clean_file']).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f'Missing original train CSV: {path}')
        # Deterministic spaced sampling; avoid loading the full dataset into RAM.
        stride = max(1, int(ep['actual_trainable_steps']) // per_ep)
        with path.open('r',encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):
                if i % stride or i // stride >= per_ep: continue
                if row['split']!='train' or int(row['seed'])!=ep['seed']:
                    raise ValueError('Original train CSV split mismatch')
                vec=features.row_features(row)
                target=float(row[features.LABEL])
                pred=features.steering_prediction(baseline,row)
                X.append(vec);y.append(target-pred);w.append(1.5)
    return X,y,w


def offline_validation(dataset, manifest, baseline, candidate, features):
    """Informational validation only, NEVER reads held-out test files."""
    root=Path(dataset).resolve(); errors_base=[]; errors_candidate=[]; maps={}
    for ep in manifest['episodes']:
        if ep['split']!='val': continue
        p=(root/ep['clean_file']).resolve()
        if root not in p.parents or not p.is_file(): raise FileNotFoundError(p)
        with p.open('r',encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                truth=float(row[features.LABEL]); b=features.steering_prediction(baseline,row)
                pred,_,_,gate=predict(candidate,row,features)
                errors_base.append(abs(truth-b));errors_candidate.append(abs(truth-pred))
                item=maps.setdefault(ep['map'],[[],[]]);item[0].append(abs(truth-b));item[1].append(abs(truth-pred))
    if not errors_base: raise ValueError('No validation rows')
    return dict(n=len(errors_base),baseline_mae=float(np.mean(errors_base)),
                candidate_mae=float(np.mean(errors_candidate)),
                by_map={m:dict(baseline_mae=float(np.mean(v[0])),candidate_mae=float(np.mean(v[1])))
                        for m,v in sorted(maps.items())},
                note='Offline imitation MAE is NOT the go/no-go criterion; use closed-loop validation')


def train(args):
    features,_ = legacy_modules(args.legacy_root)
    original,splits,manifest_hash=original_manifest(args.dataset)
    baseline=joblib.load(args.baseline) # Trusted local artifact only
    check_baseline(baseline,manifest_hash,features)
    collection_root=Path(args.collection)
    manifest_file=collection_root/'collection_manifest.json'
    manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('version')!=VERSION or manifest['config']['original_manifest_sha256']!=manifest_hash or manifest['config']['baseline_sha256']!=digest(args.baseline):
        raise ValueError('Collection is incompatible with baseline or original manifest')
    expected=len(manifest['config']['maps'])*len(manifest['config']['seeds'])*len(manifest['config']['speeds_mps'])
    if len(manifest['episodes'])!=expected:
        raise ValueError(f'Incomplete on-policy collection: {len(manifest["episodes"])}/{expected}')
    if not 0 <= args.alpha <= 1: raise ValueError('--alpha must be in [0,1]')
    X,y,w,stats=load_train_rows(collection_root,manifest,features,splits)
    X0,y0,w0=load_anchors(args.dataset,original,baseline,features)
    x=np.asarray(X+X0,dtype=np.float32);target=np.asarray(y+y0,dtype=np.float32)
    weights=np.asarray(w+w0,dtype=np.float32)
    model=ExtraTreesRegressor(n_estimators=args.trees,min_samples_leaf=5,
                              max_features=.9,n_jobs=-1,random_state=42)
    print(f'Training recovery residual on {len(X):,} DAgger + {len(X0):,} anchor rows...',flush=True)
    model.fit(x,target,sample_weight=weights)
    bundle=dict(version=VERSION,baseline=baseline,residual_model=model,alpha=args.alpha,
                teacher_fallback=False,teacher_policy_at_inference=False,
                learned_steering=True,learned_longitudinal=False,
                features=list(features.FEATURES),trained_from_collection_sha256=digest(manifest_file),
                trained_on_original_manifest_sha256=manifest_hash,
                baseline_sha256=digest(args.baseline),train_seeds=manifest['config']['seeds'],
                validation_policy='closed-loop comparison required before promotion')
    validation=offline_validation(args.dataset,original,baseline,bundle,features)
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError('Output already exists; use --overwrite or a new path')
    out.mkdir(parents=True,exist_ok=True)
    path=out/'agent4_v315_recovery.joblib'
    joblib.dump(bundle,path,compress=3)
    report=dict(version=VERSION,baseline_sha256=digest(args.baseline),
                original_manifest_sha256=manifest_hash,train=stats,anchors=len(X0),
                trees=args.trees,alpha=args.alpha,offline_validation=validation,
                note='Proposed model only: baseline remains authoritative until closed-loop validation passes. No test-set use.')
    atomic_json(out/'metrics_v315.json',report)
    print('Saved candidate:',path.resolve())
    print('Val offline baseline MAE:',round(validation['baseline_mae'],6),'candidate MAE:',round(validation['candidate_mae'],6))
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('..'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--collection',type=Path,default=Path('../agent4_v315_data'))
    p.add_argument('--out-dir',type=Path,default=Path('../agent4_v315_model'))
    p.add_argument('--trees',type=int,default=80)
    p.add_argument('--alpha',type=float,default=.6)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(argv)
    if not 16 <= args.trees <= 512: raise ValueError('--trees must be 16..512')
    return train(args)

if __name__=='__main__':main()
