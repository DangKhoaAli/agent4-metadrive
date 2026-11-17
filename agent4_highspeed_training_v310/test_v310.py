"""Offline contract + synthetic supervised-recovery tests. NOT MetaDrive driving test."""
from __future__ import annotations
import csv
import json
import tempfile
from pathlib import Path
import math
import numpy as np
from sklearn.linear_model import Ridge

from common_v310 import (VERSION,DATA_STATUS,MAX_STEER,MAX_RESIDUAL,
     basis,student_action,feature_names,validate_seed_groups,
     parse_seed_spec,parse_values,perturbation)
from train_v310 import read_data,train
from eval_v310 import patch_student_predict, v30, speed_statistics


def reject(action):
    try:action()
    except (ValueError,KeyError):return
    raise AssertionError('Expected rejection')


def test_contract():
    assert validate_seed_groups([130,131],[142],[146])=={'train':[130,131],'val':[142],'eval':[146]}
    reject(lambda:validate_seed_groups([130],[130],[146]))
    reject(lambda:validate_seed_groups([124],[142],[146]))
    assert parse_seed_spec('130-132,135')==[130,131,132,135]
    reject(lambda:parse_seed_spec('130,130'))
    assert parse_values('4,8',(4.,6.,8.,10.),float)==[4.,8.]
    reject(lambda:parse_values('4,12',(4.,6.,8.,10.),float))
    assert perturbation(45,0,.004)==.004
    assert perturbation(108,0,.004)==-.004
    assert perturbation(0,0,.004)==0
    reject(lambda:perturbation(40,0,.1))
    assert np.allclose(basis(0,0,10),0)
    assert basis(.1,-.02,8).shape==(6,)
    sample=[{'speed_mps':3.,'desired_mps':4.},{'speed_mps':4.,'desired_mps':4.}]
    assert speed_statistics(sample,2)['max_speed_mps']==4.
    reject(lambda:speed_statistics(sample,3))


def test_zero_centered():
    mod=Ridge(alpha=1e-8,fit_intercept=False).fit(
        np.array([basis(e,h,v) for e,h,v in [(.1,.01,4),(-.1,-.03,10),(.15,-.01,8),(.05,.02,6),(-.02,.03,6),(-.2,.005,4)]]),
        [.008-.003,-.008+.009,.012+.003,.004-.006,-.0016-.009,-.016-.0015])
    names=feature_names(['x','speed_mps'])
    bundle={'version':VERSION,'student_model':mod,'feature_order':names,
            'max_steer':MAX_STEER,'max_learned_residual':MAX_RESIDUAL}
    row={k:0 for k in names};row.update(speed_mps=6.,base_pdd_steer_md=.001)
    p=student_action(bundle,row)
    assert abs(p['learned_residual'])<1e-12
    assert abs(p['steer']-.001)<1e-12


def test_training_pipeline():
    # Small synthetic CSV that exercises file splitting, feature contract,
    # training and teacher imitation without claiming MetaDrive behavior.
    names=feature_names(['dummy','speed_mps'])
    fields=['map','seed','split','step','target_speed_mps','noise_md',
            'teacher_steer_md','applied_steer_md','pdd_steer_md','teacher_residual_md']+names
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        maps=['S','SSSSS'];speeds=[4.,6.,8.,10.]
        ns={'train':0,'val':0}
        for split,seeds in [('train',[130,131]),('val',[142])]:
            with (folder/f'{split}.csv').open('w',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                for m in maps:
                    for speed in speeds:
                        for seed in seeds:
                            for step in range(120):
                                e=.06*math.sin((step+seed)*.11)
                                h=.02*math.cos((step+seed)*.07)
                                v=speed*(.8+.2*step/120)
                                base=.003
                                target=base+.08*e-.3*h
                                row={k:0.0 for k in names}
                                row.update(dummy=1.,speed_mps=v,base_pdd_steer_md=base,
                                           planned_lateral_right_m=e,heading_error_rad=h)
                                writer.writerow({'map':m,'seed':seed,'split':split,'step':step,
                                    'target_speed_mps':speed,'noise_md':.0035,
                                    'teacher_steer_md':target,'applied_steer_md':target+.0035,
                                    'pdd_steer_md':base,'teacher_residual_md':target-base,**row})
                                ns[split]+=1
        manifest={'status':DATA_STATUS,'maps':maps,'speeds_mps':speeds,
            'splits':{'train':[130,131],'val':[142],'eval':[146,147]},
            'rows_by_split':ns}
        (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        result=train(folder,['dummy','speed_mps'])
        assert result['version']==VERSION
        assert result['metrics']['val']['teacher_vs_student']['mae']<.001
        assert result['metrics']['val']['teacher_vs_student']['mae'] < result['metrics']['val']['teacher_vs_pdd']['mae']
        # Reject wrong split as measured, never allow evaluation rows into training.
        altered=json.loads((folder/'manifest.json').read_text())
        altered['splits']['train']=[142]
        (folder/'manifest.json').write_text(json.dumps(altered),encoding='utf-8')
        reject(lambda:train(folder,['dummy','speed_mps']))


def test_inference_adapter():
    # Unit test that a student prediction is directly translated to applied
    # MetaDrive steering via v30 baseline, without invoking rule_lateral.
    x=np.asarray([basis(.1,-.02,8),basis(-.1,.02,8),basis(.2,.01,4)])
    y=np.asarray([.08*.1-.30*(-.02),.08*(-.1)-.30*.02,.08*.2-.30*.01])
    m=Ridge(alpha=0.,fit_intercept=False).fit(x,y)
    names=feature_names(['speed_mps'])
    bundle={'version':VERSION,'student_model':m,'feature_order':names,
            'max_steer':MAX_STEER,'max_learned_residual':MAX_RESIDUAL}
    f={'speed_mps':8.0};r={'adapted':False}
    geom={'lateral_right_m':.1,'heading_error_rad':-.02}
    old_f,old_g,old_p=v30.feature_for_model,v30.planned_geometry,v30.predict_steering
    try:
        v30.feature_for_model=lambda *_:(f,r)
        v30.planned_geometry=lambda *_:geom
        v30.predict_steering=lambda *_:-.001
        with patch_student_predict(bundle,['speed_mps']) as recorded:
            actual_f,route=v30.feature_for_model(None,2)
            assert route is r
            actual_g=v30.planned_geometry(None,2)
            carla=v30.predict_steering(None,actual_f)
            assert len(recorded)==1
            assert abs(-carla-recorded[0]['steer'])<1e-12
            assert abs(recorded[0]['base']-.001)<1e-12
    finally:
        v30.feature_for_model=old_f
        v30.planned_geometry=old_g
        v30.predict_steering=old_p


if __name__=='__main__':
    test_contract();test_zero_centered();test_training_pipeline();test_inference_adapter()
    print('Agent 4 V3.10 offline tests: PASS (synthetic fixture; not MetaDrive)')
