"""Fast standard-library unit tests: no MetaDrive needed."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from common_v315 import (VERSION, allowed_evaluation_seeds, allowed_train_seeds,
    bounded, check_baseline, episode_status, original_manifest, predict, read_step,
    recovery_gate)
from collect_v315 import noise_at_step, trainable
from eval_v315 import compare, summarize


class FakeFeatures:
    VERSION = 'LEGACY'
    FEATURES = ('f',)
    @staticmethod
    def steering_prediction(bundle,pre):return 0.1
    @staticmethod
    def row_features(pre):
        import numpy as np
        return np.array([1.],dtype='float32')


class Tests(unittest.TestCase):
    def setUp(self):
        self.split={'train':{170,171},'val':{186},'test':{190}}
        self.pre=dict(planned_lateral_right_m=0.,heading_error_rad=0.,edge_clearance_proxy_m=.8,
                      lookahead_5m_available=1,lookahead_10m_available=1)

    def test_train_only(self):
        self.assertEqual(allowed_train_seeds([170],self.split),[170])
        with self.assertRaises(ValueError):allowed_train_seeds([186],self.split)
        with self.assertRaises(ValueError):allowed_train_seeds([190],self.split)

    def test_final_seeds_new_only(self):
        self.assertEqual(allowed_evaluation_seeds([194],self.split,'final',True),[194])
        with self.assertRaises(ValueError):allowed_evaluation_seeds([190],self.split,'final',True)
        with self.assertRaises(ValueError):allowed_evaluation_seeds([194],self.split,'final',False)

    def test_validation_only(self):
        self.assertEqual(allowed_evaluation_seeds([186],self.split,'validation'),[186])
        with self.assertRaises(ValueError):allowed_evaluation_seeds([170],self.split,'validation')

    def test_gate(self):
        self.assertEqual(recovery_gate(self.pre),0)
        self.pre['planned_lateral_right_m']=.35
        self.assertAlmostEqual(recovery_gate(self.pre),1)
        self.pre['planned_lateral_right_m']=.2
        self.assertTrue(0<recovery_gate(self.pre)<1)

    def test_baseline_prediction_no_residual_near_center(self):
        model=Mock();model.predict.side_effect=AssertionError('head must not be called')
        bundle=dict(version=VERSION,baseline={},residual_model=model,teacher_fallback=False,alpha=.6)
        final,base,correction,gate=predict(bundle,self.pre,FakeFeatures)
        self.assertAlmostEqual(final,.1);self.assertEqual(gate,0)
        model.predict.assert_not_called()

    def test_recovery_residual_magnitude(self):
        self.pre['planned_lateral_right_m']=.4
        model=Mock();model.predict.return_value=[.2]
        bundle=dict(version=VERSION,baseline={},residual_model=model,teacher_fallback=False,alpha=.5)
        value,base,corr,gate=predict(bundle,self.pre,FakeFeatures)
        self.assertAlmostEqual(value,.2)
        self.assertAlmostEqual(corr,.2)

    def test_alpha_zero_identity(self):
        self.pre['planned_lateral_right_m']=.6
        model=Mock();model.predict.side_effect=AssertionError('head must not be called')
        bundle=dict(version=VERSION,baseline={},residual_model=model,teacher_fallback=False,alpha=.8)
        value,_,_,_=predict(bundle,self.pre,FakeFeatures,alpha=0)
        self.assertAlmostEqual(value,.1)

    def test_deterministic_perturb(self):
        self.assertEqual(noise_at_step(170,80,self.pre,.08),0)
        self.assertNotEqual(noise_at_step(170,95,self.pre,.08),0)
        self.pre['edge_clearance_proxy_m']=.25
        self.assertEqual(noise_at_step(170,95,self.pre,.08),0)

    def test_spike_is_discarded(self):
        reason=[]
        self.assertFalse(trainable(self.pre,1.,2.4,0.,reason))
        self.assertIn('expert_raw_out_of_range',reason)

    def test_step_interfaces(self):
        self.assertEqual(read_step([0,0,True,False,{'arrive_dest':True}]),(True,False,{'arrive_dest':True}))
        self.assertEqual(read_step([0,0,True,{'out_of_road':True}]),(True,False,{'out_of_road':True}))
        self.assertEqual(episode_status(True,False,{'out_of_road':True}),'out_of_road')

    def test_model_signature_check(self):
        with self.assertRaises(ValueError):check_baseline({'version':'bad'},'abc',FakeFeatures)
        self.assertEqual(bounded(3.),1.)

    def test_acceptance_no_regression(self):
        base=[dict(profile='baseline_v313',map='C',seed=186,speed_mps=2,confirmed_success=True,quality_pass=False,status='arrive_dest'),
              dict(profile='baseline_v313',map='C',seed=186,speed_mps=3,confirmed_success=True,quality_pass=True,status='arrive_dest')]
        candidate=[dict(profile='candidate_v315',map='C',seed=186,speed_mps=2,confirmed_success=True,quality_pass=True,status='arrive_dest'),
                   dict(profile='candidate_v315',map='C',seed=186,speed_mps=3,confirmed_success=True,quality_pass=True,status='arrive_dest')]
        self.assertTrue(compare(base+candidate,'validation')['accepted'])
        candidate[1]['quality_pass']=False
        self.assertFalse(compare(base+candidate,'validation')['accepted'])

    def test_dataset_manifest_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            doc=dict(version='AGENT4_V312B_FULL_STEP_IDM_DATASET',
                     config=dict(split_seeds={'train':[170],'val':[186],'test':[190]}))
            (root/'manifest.json').write_text(json.dumps(doc),encoding='utf-8')
            _,splits,sha=original_manifest(root)
            self.assertEqual(len(sha),64)
            self.assertEqual(splits['train'],{170})


if __name__=='__main__':unittest.main(verbosity=2)
