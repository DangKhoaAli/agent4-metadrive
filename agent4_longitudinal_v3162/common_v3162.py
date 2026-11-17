"""Shared contracts for V3.16.2 on-policy longitudinal DAgger.

All trained actions come from an estimator. The frozen V3.13 steering is unchanged.
No IDM or rule governor is used at candidate inference.
"""
from __future__ import annotations
import math,sys
from pathlib import Path
import numpy as np

SIBLING=Path(__file__).resolve().parent.parent/'agent4_longitudinal_v316'
if not (SIBLING/'common_v316.py').is_file():
    raise FileNotFoundError('Unpack V3.16.2 beside agent4_longitudinal_v316')
if str(SIBLING) not in sys.path:sys.path.insert(0,str(SIBLING))
from common_v316 import (dependencies,sha,speed_schedule,make_features,model_features,
    predict_long,validate_model,allowable_seeds,pair_action,verified_action)

VERSION='AGENT4_V3162_ON_POLICY_LONGITUDINAL_DAGGER'

def validate_v3162(bundle,original_hash,baseline_hash,actor_hash,feat):
    if bundle.get('version')!=VERSION or bundle.get('teacher_fallback') is not False:
        raise ValueError('Invalid V3.16.2 estimator bundle')
    if bundle.get('original_manifest_sha256')!=original_hash or bundle.get('baseline_sha256')!=baseline_hash:
        raise ValueError('Original dataset/steering mismatch')
    if bundle.get('actor_sha256')!=actor_hash:
        raise ValueError('V3.16 actor differs from training provenance')
    if tuple(bundle.get('feature_names',()))!=model_features(feat):
        raise ValueError('Unexpected feature schema')
    if bundle.get('trained_on_split')!='train':
        raise ValueError('Model not trained strictly on TRAIN')

def predict_v3162(bundle,row,feat):
    if bundle.get('version')!=VERSION or bundle.get('teacher_fallback') is not False:
        raise ValueError('Unexpected V3.16.2 model')
    if tuple(bundle.get('feature_names',()))!=model_features(feat):
        raise ValueError('Feature mismatch')
    value=float(bundle['estimator'].predict(make_features(row,feat).reshape(1,-1))[0])
    if not math.isfinite(value):raise ValueError('NaN model action')
    return float(np.clip(value,-1.0,1.0))
