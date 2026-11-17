"""Offline contracts: no MetaDrive needed. Does not claim simulator validation."""
from __future__ import annotations
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common_v316 import (VERSION,EXTRA,USED_PRIOR_FINAL,allowable_seeds,
    make_features,model_features,pair_action,predict_long,speed_schedule,validate_model,verified_action)
from collect_v316 import label_quality,set_shadow_target
from eval_v316 import PROFILES,comparison,summarize

class FakeFeatures:
    FEATURES=('speed_mps','planned_lateral_right_m','heading_error_rad','lookahead_5m_available','lookahead_10m_available')
    def row_features(self,row):return np.array([float(row[k]) for k in self.FEATURES],dtype=np.float32)

class FakePredictor:
    def predict(self,x):
        assert x.shape[1]==7
        return np.array([-0.12])

class FakeCar:
    def __init__(self):self.current_action=[.2,-.1];self.steering=.2;self.throttle_brake=-.1

class FakeShadow:
    NORMAL_SPEED=0; CREEP_SPEED=0;target_speed=0

def good_row():
    return dict(speed_mps=2.,planned_lateral_right_m=.01,heading_error_rad=.01,
                lookahead_5m_available=1,lookahead_10m_available=1,
                target_speed_mps=1.,edge_clearance_proxy_m=.8)

def e(profile,quality=True,arrival=True,mae=.2,brakes=10):
    return dict(profile=profile,map='C',seed=186,speed_mps=3.,
                confirmed_success=arrival,quality_pass=quality,status='arrive_dest' if arrival else 'out_of_road',
                speed_mae_mps=mae,steps=100,model_brake_steps=brakes,overspeed_steps=0)

class Contracts(unittest.TestCase):
    def test_schedule_start_cruise(self):self.assertEqual(speed_schedule(4,64),4)
    def test_schedule_slow(self):self.assertEqual(speed_schedule(4,65),2.2)
    def test_schedule_restores(self):self.assertEqual(speed_schedule(4,120),4)
    def test_schedule_repeats(self):self.assertEqual(speed_schedule(4,245),2.2)
    def test_schedule_rejects_nan(self):
        with self.assertRaises(ValueError):speed_schedule(float('nan'),2)
    def test_split_train(self):self.assertEqual(allowable_seeds([170],{'train':{170},'val':{186},'test':{190}},'train'),[170])
    def test_split_leakage(self):
        with self.assertRaises(ValueError):allowable_seeds([186],{'train':{170},'val':{186},'test':{190}},'train')
    def test_final_previous_banned(self):
        with self.assertRaises(ValueError):allowable_seeds([201],{'train':{170},'val':{186},'test':{190}},'final')
    def test_final_fresh(self):self.assertEqual(allowable_seeds([202],{'train':{170},'val':{186},'test':{190}},'final'),[202])
    def test_features_speed_error(self):
        r=make_features(good_row(),FakeFeatures())
        self.assertEqual(len(r),7);self.assertEqual(float(r[-1]),-1.)
    def test_valid_label(self):self.assertEqual(label_quality(good_row(),-.4),(True,''))
    def test_invalid_raw_label(self):self.assertFalse(label_quality(good_row(),-2)[0])
    def test_invalid_lookahead(self):
        row=good_row();row['lookahead_10m_available']=0
        self.assertFalse(label_quality(row,-.4)[0])
    def test_shadow_target_kmh(self):
        policy=FakeShadow();set_shadow_target(policy,3)
        self.assertAlmostEqual(policy.target_speed,10.8)
    def test_pair_invalid(self):
        with self.assertRaises(ValueError):pair_action([1.,float('nan')])
    def test_applied_controls(self):self.assertEqual(verified_action(FakeCar(),.2,-.1),(.2,-.1))
    def test_applied_mismatch(self):
        with self.assertRaises(RuntimeError):verified_action(FakeCar(),.3,-.1)
    def test_predicted_direct_signed(self):
        bundle=dict(version=VERSION,teacher_fallback=False,
            feature_names=list(model_features(FakeFeatures())),estimator=FakePredictor())
        self.assertAlmostEqual(predict_long(bundle,good_row(),FakeFeatures()),-.12)
    def test_no_teacher_fallback(self):
        bundle=dict(version=VERSION,teacher_fallback=True,
            feature_names=list(model_features(FakeFeatures())),estimator=FakePredictor())
        with self.assertRaises(ValueError):predict_long(bundle,good_row(),FakeFeatures())
    def test_model_integrity(self):
        b=dict(version=VERSION,teacher_fallback=False,feature_names=list(model_features(FakeFeatures())),
               original_manifest_sha256='abc',baseline_sha256='xyz')
        validate_model(b,'abc','xyz',FakeFeatures())
        with self.assertRaises(ValueError):validate_model(b,'other','xyz',FakeFeatures())
    def test_comparison_accepted(self):
        rows=[e(PROFILES[0],mae=.2),e(PROFILES[1],mae=.21)]
        self.assertTrue(comparison(rows,'validation')['accepted'])
        self.assertFalse(comparison(rows,'final')['accepted'])
    def test_comparison_regression_rejected(self):
        rows=[e(PROFILES[0],mae=.2),e(PROFILES[1],quality=False,mae=.1)]
        self.assertFalse(comparison(rows,'validation')['accepted'])
    def test_comparison_speed_regression_rejected(self):
        rows=[e(PROFILES[0],mae=.2),e(PROFILES[1],mae=.5)]
        self.assertFalse(comparison(rows,'validation')['accepted'])
    def test_comparison_no_brakes_rejected(self):
        rows=[e(PROFILES[0],mae=.2),e(PROFILES[1],mae=.2,brakes=0)]
        self.assertFalse(comparison(rows,'validation')['accepted'])
    def test_comparison_incomplete_rejected(self):
        self.assertFalse(comparison([e(PROFILES[0])],'validation')['accepted'])

if __name__=='__main__':unittest.main(verbosity=2)
