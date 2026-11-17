"""V3.16.2 offline safety/contract tests. Does not require a MetaDrive runtime."""
from __future__ import annotations
import unittest,tempfile,csv,json,sys,copy
from pathlib import Path
import numpy as np
from common_v3162 import VERSION,predict_v3162,validate_v3162
from train_v3162 import read_dagger
from eval_v3162 import RULE,RAW,NEW,compare,summary_for,load_reference

class DummyFeat:
    FEATURES=('speed_mps',)
    @staticmethod
    def row_features(row):return np.array([float(row['speed_mps'])])
class Constant:
    def __init__(self,y):self.y=y
    def predict(self,x):return np.full(x.shape[0],self.y)

BASE=dict(version=VERSION,teacher_fallback=False,
          original_manifest_sha256='orig',baseline_sha256='base',actor_sha256='actor',
          feature_names=['speed_mps','target_speed_mps','speed_error_mps'],trained_on_split='train',
          estimator=Constant(.03))

def episode(profile,m='S',seed=186,speed=2.0,quality=True,arrived=True,mae=.1):
    return dict(profile=profile,map=m,seed=seed,speed_mps=speed,
         status='arrive_dest' if arrived else 'out_of_road',confirmed_success=arrived,
         quality_pass=quality,steps=100,verified_steps=100,speed_mae_mps=mae,
         overspeed_steps=0,model_brake_steps=20,errors=[],teacher_calls=0,idm_policy_used=False)

class BundleTests(unittest.TestCase):
    def test_valid_bundle(self):validate_v3162(BASE,'orig','base','actor',DummyFeat)
    def test_wrong_bundle_version(self):
        with self.assertRaises(ValueError):validate_v3162({**BASE,'version':'v316'},'orig','base','actor',DummyFeat)
    def test_no_teacher_fallback(self):
        with self.assertRaises(ValueError):validate_v3162({**BASE,'teacher_fallback':True},'orig','base','actor',DummyFeat)
    def test_wrong_original_hash(self):
        with self.assertRaises(ValueError):validate_v3162(BASE,'X','base','actor',DummyFeat)
    def test_wrong_baseline_hash(self):
        with self.assertRaises(ValueError):validate_v3162(BASE,'orig','X','actor',DummyFeat)
    def test_wrong_actor_hash(self):
        with self.assertRaises(ValueError):validate_v3162(BASE,'orig','base','X',DummyFeat)
    def test_wrong_schema(self):
        with self.assertRaises(ValueError):validate_v3162({**BASE,'feature_names':['oops']},'orig','base','actor',DummyFeat)
    def test_no_val_training(self):
        with self.assertRaises(ValueError):validate_v3162({**BASE,'trained_on_split':'val'},'orig','base','actor',DummyFeat)
    def test_predict_model_output(self):
        self.assertAlmostEqual(predict_v3162(BASE,dict(speed_mps=1.,target_speed_mps=2.),DummyFeat),.03)
    def test_model_predict_clipped(self):
        self.assertEqual(predict_v3162({**BASE,'estimator':Constant(1.8)},dict(speed_mps=1.,target_speed_mps=2.),DummyFeat),1.0)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):predict_v3162({**BASE,'estimator':Constant(float('nan'))},dict(speed_mps=1.,target_speed_mps=2.),DummyFeat)

class CompareTests(unittest.TestCase):
    def test_subset_cannot_promote(self):
        e=[episode(p) for p in (RULE,RAW,NEW)]
        self.assertFalse(compare(e,'validation')['accepted'])
    def test_detect_quality_regression_vs_rule(self):
        e=[episode(RULE,quality=True),episode(RAW,quality=False),episode(NEW,quality=False)]
        self.assertTrue(any('quality_vs_rule' in r for r in compare(e,'validation')['regressions']))
    def test_detect_quality_regression_vs_raw(self):
        e=[episode(RULE,quality=False),episode(RAW,quality=True),episode(NEW,quality=False)]
        self.assertTrue(any('quality_vs_learned_longitudinal_v316' in r for r in compare(e,'validation')['regressions']))
    def test_detect_arrival_regression(self):
        e=[episode(RULE),episode(RAW),episode(NEW,arrived=False,quality=False)]
        self.assertTrue(any('arrival_vs' in r for r in compare(e,'validation')['regressions']))
    def test_reject_duplicate(self):
        with self.assertRaises(ValueError):compare([episode(RULE),episode(RULE)],'validation')
    def test_full_good_promotes(self):
        e=[episode(p,m,s,v,mae=.1) for m in 'SCX' for s in range(186,190) for v in (2.,3.,4.) for p in (RULE,RAW,NEW)]
        d=compare(e,'validation')
        self.assertTrue(d['full_validation_suite']);self.assertTrue(d['accepted'])
    def test_final_never_promotes(self):
        e=[episode(p,m,s,v) for m in 'SCX' for s in range(186,190) for v in (2.,3.,4.) for p in (RULE,RAW,NEW)]
        self.assertFalse(compare(e,'final')['accepted'])
    def test_speed_regression_rejected(self):
        e=[episode(p,m,s,v,mae=.3 if p==NEW else .1) for m in 'SCX' for s in range(186,190) for v in (2.,3.,4.) for p in (RULE,RAW,NEW)]
        self.assertFalse(compare(e,'validation')['speed_tracking_within_tolerance'])
    def test_missing_reference_rejected(self):
        e=[episode(RULE),episode(NEW)]
        self.assertFalse(compare(e,'validation')['complete'])
    def test_missing_candidate_rejected(self):
        e=[episode(RULE),episode(RAW)]
        self.assertFalse(compare(e,'validation')['accepted'])
    def test_zero_braking_rejected(self):
        e=[episode(p,m,s,v) for m in 'SCX' for s in range(186,190) for v in (2.,3.,4.) for p in (RULE,RAW,NEW)]
        for item in e:
            if item['profile']==NEW:item['model_brake_steps']=0
        self.assertFalse(compare(e,'validation')['braking_observed'])
    def test_summary_weighted(self):
        e=[episode(RULE,mae=.15),episode(RAW,mae=.2),episode(NEW,mae=.05)]
        self.assertAlmostEqual(summary_for(e)[NEW]['speed_mae_mps'],.05)

class DaggerDataTests(unittest.TestCase):
    def test_leakage_rejected(self):
        doc={'config':{'train_seeds':[186],'maps':['C'],'speeds_mps':[2],
            'behavior_policy':'frozen V3.16 gas/brake + V3.13 steering'},'episodes':[]}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):read_dagger(Path(td),doc,DummyFeat,{'train':set(range(170,186))})
    def test_wrong_behavior_policy(self):
        doc={'config':{'train_seeds':[170],'maps':['C'],'speeds_mps':[2],
             'behavior_policy':'rule governor'},'episodes':[]}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):read_dagger(Path(td),doc,DummyFeat,{'train':{170}})
    def test_incomplete_episodes_rejected(self):
        doc={'config':{'train_seeds':[170],'maps':['C'],'speeds_mps':[2],
             'behavior_policy':'frozen V3.16 gas/brake + V3.13 steering'},'episodes':[]}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):read_dagger(Path(td),doc,DummyFeat,{'train':{170}})

class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=Path(__file__).resolve().parent.parent/'agent4_v316_validation.json'
    def test_reference_full(self):
        if not self.f.exists():self.skipTest('Prior validation file not present in local source tree')
        doc=json.loads(self.f.read_text())
        cfg=doc['config']
        rows=load_reference(self.f,cfg['model_sha256'],cfg['baseline_sha256'],['C'],[186,189],[2.],cfg['steps'])
        self.assertEqual(len(rows),4)
    def test_reference_wrong_actor_hash(self):
        if not self.f.exists():self.skipTest('Prior validation file not present in local source tree')
        cfg=json.loads(self.f.read_text())['config']
        with self.assertRaises(ValueError):load_reference(self.f,'WRONG',cfg['baseline_sha256'],['C'],[186],[2.],cfg['steps'])

if __name__=='__main__':unittest.main(verbosity=2)
