"""Offline sanity tests; no MetaDrive installation required."""
from __future__ import annotations
import csv
import json
import tempfile
import sys
import types
from unittest.mock import patch
from pathlib import Path
import numpy as np
from common_v33 import (feature_names,state_features,student_steer_md,
                        seed_sets,clip,MAX_STEER,MAX_STUDENT_RESIDUAL)
from collect_teacher_v33 import steering_perturbation,seeds,make_slow_idm_policy
from train_student_v33 import train
from eval_student_v33 import checked_student_bundle

class ConstantPredictor:
    def __init__(self,val):self.val=val
    def predict(self,x):return np.full((len(x),),self.val)


def test_contract():
    pdd=['speed_mps','wp1_right_m']
    state=state_features({'speed_mps':3,'wp1_right_m':-4},pdd,-.08,0.3,.11,False)
    assert list(state)==feature_names(pdd)
    a=student_steer_md(ConstantPredictor(-.4),state,list(state))
    assert abs(a['learned_residual']+MAX_STUDENT_RESIDUAL)<1e-9
    assert abs(a['steer']+.33)<1e-9
    assert clip(2,MAX_STEER)==MAX_STEER
    assert seeds('4-6,9')==[4,5,6,9]
    assert seed_sets([4,5],[6],[7])['eval']==[7]
    try:seed_sets([4,5],[5],[7])
    except ValueError:pass
    else:raise AssertionError('Data leakage was accepted')
    assert steering_perturbation(50,0,.012)==.012
    assert steering_perturbation(120,0,.012)==-.012
    assert steering_perturbation(3,0,.012)==0
    try:steering_perturbation(10,0,.12)
    except ValueError:pass
    else:raise AssertionError('Unsafe perturbation accepted')


def test_recorded_idm_policy_with_stub():
    # Verify the slow-expert action alignment/label/perturbation logic without MetaDrive.
    base=types.ModuleType('metadrive')
    policy_package=types.ModuleType('metadrive.policy')
    module=types.ModuleType('metadrive.policy.idm_policy')
    class IDMPolicyStub:
        def __init__(self,*args,**kwargs):pass
        def act(self,*args,**kwargs):return [0.42,0.75]
    module.IDMPolicy=IDMPolicyStub
    with patch.dict(sys.modules,{'metadrive':base,'metadrive.policy':policy_package,
                                  'metadrive.policy.idm_policy':module}):
        Klass=make_slow_idm_policy()
        expert=Klass()
        expert.queue_action(0.07,-0.012)
        applied=expert.act()
        assert expert.record_count==1
        assert abs(expert.last_record['native_steer']-0.42)<1e-9
        assert abs(expert.last_record['teacher_steer']-0.35)<1e-9
        assert abs(expert.last_record['applied_steer']-0.338)<1e-9
        assert all(abs(a-b)<1e-9 for a,b in zip(applied,[0.338,0.07]))


def test_train_synthetic():
    pdd=['speed_mps','wp1_right_m']
    names=feature_names(pdd)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        spec=[('train',4,520),('val',24,180)]
        for name,seed,n in spec:
            with (root/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=['map','seed','teacher_steer_md']+names)
                w.writeheader()
                for i in range(n):
                    rng=(i%40)/40
                    state=state_features({'speed_mps':3.0,'wp1_right_m':rng*3},pdd,
                                         -.06+0.1*rng,0.25*rng,.05*rng,False)
                    steer=clip(float(state['base_pdd_steer_md'])+0.08*rng-0.02,MAX_STEER)
                    w.writerow({'map':'C','seed':seed,'teacher_steer_md':steer,**state})
        manifest={'splits':{'train':[4],'val':[24],'eval':[28,29]}}
        path=root/'manifest.json'
        path.write_text(json.dumps(manifest),encoding='utf-8')
        out=train(root/'train.csv',root/'val.csv',pdd,path,max_iter=30)
        assert list(out['feature_order'])==names
        assert out['metrics']['val']['learned_student']['n']==180
        assert out['metrics']['val']['student_better_mae']
        assert checked_student_bundle(out,pdd,[28,29])
        try:checked_student_bundle(out,pdd,[24])
        except ValueError:pass
        else:raise AssertionError('Closed-loop data leakage undetected')
    print('V3.3 feature, splits, synthetic teacher training and leakage tests: PASS')

if __name__=='__main__':
    test_contract()
    test_recorded_idm_policy_with_stub()
    test_train_synthetic()
