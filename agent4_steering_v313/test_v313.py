"""Offline unit tests. Synthetic data is for code tests ONLY, not driving claims."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import math

from features_v313 import FEATURES,row_features,steering_prediction,VERSION
from train_v313 import manifest_load,load_samples,run
from eval_closed_loop_v313 import governor,run_one


def synthetic_dataset(root):
    root=Path(root)
    episodes=[]
    splits={'train':[170,171],'val':[186],'test':[190]}
    for split,seeds in splits.items():
        for m in ('S','C'):
            for seed in seeds:
                rows=[]
                for j in range(40):
                    lat=.11*math.sin(j/5.+seed) if m=='C' else .04*math.sin(j/5.)
                    heading=.03*math.cos(j/6.) if m=='C' else .008*math.cos(j/6.)
                    curve=-.015 if m=='C' else 0.
                    row={
                        'speed_mps':'3.0',
                        'planned_lateral_right_m':str(lat),
                        'heading_error_rad':str(heading),
                        'distance_to_lane_end_m':str(30.-j*.2),
                        'curvature_proxy_rad_per_m':str(curve),
                        'lookahead_heading_5m_rad':str(curve*5),
                        'lookahead_heading_10m_rad':str(curve*10),
                        'lookahead_heading_20m_rad':str(curve*20),
                        'lookahead_5m_available':'1',
                        'lookahead_10m_available':'1',
                        'lookahead_20m_available':'1',
                        'is_circular_lane':str(int(m=='C')),
                        'applied_steering':str(-2.*lat-1.5*heading+1.1*curve),
                        'trainable_row':'1','current_action_verified':'1','label_valid':'1',
                        'split':split,'map':m,'seed':str(seed),'target_speed_mps':'3.0',
                    }
                    rows.append(row)
                rel=f'clean/{split}/{m}/seed{seed}_speed3p0mps.csv'
                full=root/rel
                full.parent.mkdir(parents=True,exist_ok=True)
                with full.open('w',encoding='utf8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                episodes.append(dict(split=split,map=m,seed=seed,speed_mps=3.,
                    quality_pass=True,status='arrive_dest',recorded_actions_aligned=True,
                    errors=[],actual_trainable_steps=len(rows),clean_file=rel))
    total={s:{'clean_steps':sum(e['actual_trainable_steps'] for e in episodes if e['split']==s)} for s in splits}
    doc={'version':'AGENT4_V312B_FULL_STEP_IDM_DATASET','config':{
       'split_seeds':splits,'maps':['S','C'],'speeds_mps':[3.]},
       'episodes':episodes,'totals':{'by_split':total}}
    (root/'manifest.json').write_text(json.dumps(doc),encoding='utf8')
    return doc


class UnitTests(unittest.TestCase):
    def test_features_no_leakage(self):
        banned={'seed','split','map','pos_x_m','post_speed_mps','raw_steering',
                'applied_steering','applied_throttle_brake'}
        self.assertTrue(banned.isdisjoint(set(FEATURES)))

    def test_missing_lookahead20_explicit(self):
        pre=dict.fromkeys(FEATURES,0.)
        pre.update(lookahead_5m_available=1,lookahead_10m_available=1,
                   lookahead_20m_available=0,lookahead_heading_20m_rad=None)
        v=row_features(pre);self.assertEqual(len(v),len(FEATURES))
        self.assertEqual(v[FEATURES.index('lookahead_heading_20m_rad')],0.)

    def test_missing_when_available_rejected(self):
        pre=dict.fromkeys(FEATURES,0.)
        pre.update(lookahead_5m_available=1,lookahead_10m_available=1,
                   lookahead_20m_available=1,lookahead_heading_20m_rad=None)
        with self.assertRaises(ValueError):row_features(pre)

    def test_governor(self):
        self.assertAlmostEqual(governor(0,3),.4)
        self.assertLess(governor(4,3),0)

    def test_closed_loop_model_only_fake_env(self):
        class ConstantEstimator:
            def predict(self,X):return [0.05]
        class FakeEnv:
            def __init__(self):self.agent=SimpleNamespace(current_action=[0.,0.])
            def reset(self,seed):return None
            def step(self,action):
                self.agent.current_action=list(action)
                return None,0.,True,False,{'arrive_dest':True,'out_of_road':False,
                                        'on_yellow_continuous_line':False}
        pre=dict.fromkeys(FEATURES,0.)
        pre.update(speed_mps=2.9,lookahead_5m_available=1,
                   lookahead_10m_available=1,lookahead_20m_available=1,
                   edge_clearance_proxy_m=.8)
        bundle={'version':VERSION,'features':list(FEATURES),'estimator':ConstantEstimator()}
        with patch('eval_closed_loop_v313.state_before_action',return_value=pre):
            ep=run_one(FakeEnv(),bundle,'C',186,3.)
        self.assertTrue(ep['confirmed_success'])
        self.assertTrue(ep['quality_pass'])
        self.assertEqual(ep['teacher_fallback_calls'],0)
        self.assertEqual(ep['n_verified_prediction_steps'],1)


    def test_complete_model_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'dataset';root.mkdir()
            synthetic_dataset(root)
            doc,sha=manifest_load(root)
            ds=load_samples(root,doc)
            self.assertEqual(len(ds['train']['y']),160)
            out=Path(td)/'model'
            report=run(SimpleNamespace(dataset=root,models='ridge',out_dir=out,overwrite=False))
            self.assertEqual(report['selected_model'],'ridge')
            self.assertTrue((out/'agent4_v313_steering.joblib').is_file())
            import joblib
            b=joblib.load(out/'agent4_v313_steering.joblib')
            self.assertEqual(b['version'],VERSION)
            x={k:0. for k in FEATURES};x.update(lookahead_5m_available=1,
                    lookahead_10m_available=1,lookahead_20m_available=1)
            self.assertLessEqual(abs(steering_prediction(b,x)),1.)

    def test_reject_seed_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);doc=synthetic_dataset(root)
            doc['config']['split_seeds']['test']=[170]
            (root/'manifest.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'Seed leakage'):manifest_load(root)

    def test_reject_incomplete_data(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);doc=synthetic_dataset(root)
            doc['episodes']=doc['episodes'][:-1]
            (root/'manifest.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'incomplete'):manifest_load(root)

    def test_reject_missing_csv(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);doc=synthetic_dataset(root)
            file=root/doc['episodes'][0]['clean_file'];file.unlink()
            with self.assertRaises(FileNotFoundError):load_samples(root,doc)

if __name__=='__main__':unittest.main(verbosity=2)
