"""Offline unit tests. No MetaDrive required; cannot establish real-world performance."""
from __future__ import annotations
import csv,json,math,tempfile,sys
from pathlib import Path
import numpy as np
import collect_v37 as col
import train_v37 as tr
import eval_v37 as ev
from common_v37 import VERSION,DATA_STATUS,names,make_row,validate_new_splits

PDD=['speed_mps']+[f'feature_{i}' for i in range(14)]

def raises(error,fn,*args,**kwargs):
    try:fn(*args,**kwargs)
    except error:return
    raise AssertionError(f'Expected {error.__name__}')

def test_splits():
    groups=validate_new_splits({'train':[80,81], 'val':[92], 'eval':[96]})
    assert groups['train']==[80,81]
    raises(ValueError,validate_new_splits,{'train':[79],'val':[92],'eval':[96]})
    raises(ValueError,validate_new_splits,{'train':[80],'val':[80],'eval':[96]})
    assert names(PDD)[-1]=='speed_error_mps'

def test_teacher_capture_restores_and_aligns():
    v30=col.v30
    old=(v30.feature_for_model,v30.planned_geometry,v30.governed_speed_action)
    f={x:0. for x in PDD};f['speed_mps']=3.8
    r={'adapted':False,'remaining_m':60}
    g={'lateral_right_m':.1,'heading_error_rad':.01}
    try:
        v30.feature_for_model=lambda env,gap:(f,r)
        v30.planned_geometry=lambda agent,gap:g
        v30.governed_speed_action=lambda speed,base,remaining:(.02,base)
        cached=(v30.feature_for_model,v30.planned_geometry,v30.governed_speed_action)
        entries=[]
        with col.capture_governor_teacher(PDD,entries):
            v30.feature_for_model(type('E',(),{'agent':type('C',(),{'lane':None})()})(),2)
            v30.planned_geometry(None,2)
            act,target=v30.governed_speed_action(3.8,4,60)
            assert act==.02 and target==4
        assert entries[0]['teacher_longitudinal']==entries[0]['applied_longitudinal']==.02
        assert entries[0]['teacher_source']=='v35_governor_under_v35_steering'
        assert entries[0]['speed_error_mps']==.2 or abs(entries[0]['speed_error_mps']-.2)<1e-8
        assert (v30.feature_for_model,v30.planned_geometry,v30.governed_speed_action)==cached
        try:
            with col.capture_governor_teacher(PDD,[]):
                raise RuntimeError('test')
        except RuntimeError: pass
        assert (v30.feature_for_model,v30.planned_geometry,v30.governed_speed_action)==cached
    finally:
        (v30.feature_for_model,v30.planned_geometry,v30.governed_speed_action)=old


def fake_dataset(root):
    splits=validate_new_splits({'train':[80,81],'val':[92,93],'eval':[96,97]})
    fields=col.META+names(PDD)
    counts={'train':0,'val':0};cell={}
    for split in ('train','val'):
        with (root/f'{split}.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for m in ['S','C','X']:
                for speed in [2.,3.,4.]:
                    for seed in splits[split]:
                        for step in range(48):
                            desired=speed if step%12<7 else speed*.55
                            actual=desired + (.45 if step%4<2 else -.3)
                            features={k:0.0 for k in PDD};features['speed_mps']=actual
                            model_row=make_row(features,PDD,{'lateral_right_m':.03,'heading_error_rad':.01},False,desired,50)
                            expert=float(np.clip((desired-actual)*.15,-1,1))
                            w.writerow({'map':m,'seed':seed,'step':step,'base_speed_mps':speed,
                                'teacher_source':'v35_governor_under_v35_steering',
                                'teacher_longitudinal':expert,'applied_longitudinal':expert,
                                'longitudinal_clipped':0,'lane_class':'StraightLane',**model_row})
                            counts[split]+=1
    manifest={'status':DATA_STATUS,'splits':splits,'maps':['S','C','X'],
              'speeds_mps':[2.,3.,4.],'data_fields':fields,'rows_by_split':counts}
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')


def test_training_and_data_leakage():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);fake_dataset(root)
        model=tr.fit(root,PDD,55)
        assert model['version']==VERSION
        assert model['teacher_label'].lower().find('not idm')>=0
        assert math.isfinite(model['metrics']['val']['overall']['mae'])
        assert model['metrics']['val']['overall']['brake_n']>10
        assert model['metrics']['val']['overall']['throttle_n']>10
        with (root/'val.csv').open('a',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=col.META+names(PDD))
            w.writerow({'map':'S','seed':80,'step':1000,'base_speed_mps':2,
                        'teacher_source':'v35_governor_under_v35_steering',
                        'teacher_longitudinal':.2,'applied_longitudinal':.2,
                        'longitudinal_clipped':0,'lane_class':'StraightLane',
                        **{k:0 for k in names(PDD)}})
        raises(ValueError,tr.fit,root,PDD,55)


def test_eval_bundle_validation():
    class Dummy:
        def predict(self,x):return [0.]
    pdd={'label_mode':'at_t','features':PDD}
    steering={'version':'V35_MULTISPEED_IDM_DISTILLED_RESIDUAL',
              'feature_order':PDD+['base_pdd_steer_md','planned_lateral_right_m','heading_error_rad','adaptive_lookahead'],
              'train_seed_groups':[('S',40)],'val_seed_groups':[('S',52)],'student_model':Dummy()}
    old={'version':ev.v36.VERSION,'feature_order':names(PDD),'pdd_features':PDD,
         'splits':{'train':[60],'val':[72],'eval':[76]},'speed_targets':[2.,3.,4.]}
    new={'version':VERSION,'feature_order':names(PDD),'pdd_features':PDD,
         'splits':{'train':[80],'val':[92],'eval':[96]},'speed_targets':[2.,3.,4.]}
    ev.validate_models(pdd,steering,old,new,[96],[4.])
    raises(ValueError,ev.validate_models,pdd,steering,old,new,[80],[4.])
    raises(ValueError,ev.validate_models,pdd,steering,old,new,[96],[5.])


def main():
    for test in [test_splits,test_teacher_capture_restores_and_aligns,
                 test_training_and_data_leakage,test_eval_bundle_validation]:
        test();print('PASS',test.__name__)
    print('ALL V3.7 OFFLINE TESTS PASS. MetaDrive closed-loop has NOT been run.')

if __name__=='__main__':main()
