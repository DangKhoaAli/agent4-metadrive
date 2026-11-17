"""Offline V3.6 smoke/unit tests: no MetaDrive binary or vehicle needed."""
from __future__ import annotations
import csv,json,math,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common_v36 import (names,make_row,speed_schedule,check_splits,check_eval_seeds,
                        predict_action,VERSION)
import collect_longitudinal_v36 as collector
import train_longitudinal_v36 as trainer
import eval_longitudinal_v36 as evaluator

PDD=list(collector.v30.FEATURES)


def raises(expected,fn,*args):
    try:fn(*args)
    except expected:return
    raise AssertionError(f'Expected {expected.__name__}: {fn.__name__}')


def test_contracts():
    assert speed_schedule(4,0)==4
    assert abs(speed_schedule(4,91)-2.2)<1e-10
    assert speed_schedule(4,161)==4
    assert speed_schedule(2,100,False)==2
    groups=check_splits([60,61],[72],[76,77])
    raises(ValueError,check_splits,[60],[60],[76])
    raises(ValueError,check_splits,[40],[72],[76])
    check_eval_seeds({'splits':groups},[76])
    raises(ValueError,check_eval_seeds,{'splits':groups},[60])
    raises(ValueError,check_eval_seeds,{'splits':groups},[45])
    feat={name:0. for name in PDD}
    feat['speed_mps']=3.
    g={'lateral_right_m':.1,'heading_error_rad':.01}
    row=make_row(feat,PDD,g,False,2.5,30)
    assert list(row)==names(PDD)
    assert row['speed_error_mps']==-.5
    raises(ValueError,make_row,feat,PDD,g,False,float('nan'),30)
    class Constant:
        def predict(self,x):return [1.8]
    out=predict_action({'feature_order':names(PDD),'model':Constant()},row)
    assert out==1.


def test_recorded_policy_contract():
    class FakeIDM:
        def __init__(self,*args,**kwargs):
            self.action_info={}
            self.NORMAL_SPEED=30
            self.CREEP_SPEED=5
            self.target_speed=self.NORMAL_SPEED
        def act(self,*args,**kwargs):
            return [.12,1-(self.speed_kmh/self.target_speed)**10]
    fake_module=types.ModuleType('metadrive.policy.idm_policy')
    fake_module.IDMPolicy=FakeIDM
    with patch.dict(sys.modules,{'metadrive.policy.idm_policy':fake_module}):
        Cls=collector.make_recorded_idm_policy()
        policy=Cls(None,0)
        policy.speed_kmh=11.
        policy.set_desired_speed(2.0)
        steer,long=policy.act()
        assert policy.record_count==1
        assert long==-1.0
        assert steer==.12
        assert policy.last_record['longitudinal_clipped']==1
        assert policy.last_record['teacher_longitudinal']==policy.last_record['applied_longitudinal']
        policy.speed_kmh=4.0
        policy.set_desired_speed(3.0)
        _,long=policy.act()
        assert long>0.


def fake_dataset(root):
    f=names(PDD)
    fields=collector.CSV_META+f
    groups=check_splits([60,61],[72,73],[76,77])
    counts={}
    brake={}
    for split in ('train','val'):
        with (root/f'{split}.csv').open('w',encoding='utf-8',newline='') as fp:
            w=csv.DictWriter(fp,fieldnames=fields);w.writeheader()
            counts[split]=0
            for m in ('S','C','X'):
                for base in (2.,3.,4.):
                    for seed in groups[split]:
                        for step in range(32):
                            desired=base*(.6 if (step%8)<4 else 1.)
                            actual=desired+(.4 if step%4<2 else -.2)
                            inputs={k:0. for k in PDD}
                            inputs.update(speed_mps=actual,wp1_forward_m=25,wp2_forward_m=45,
                                          lane_width_m=3.5,front_vehicle_dist_m=100.)
                            row=make_row(inputs,PDD,{'lateral_right_m':0,'heading_error_rad':0},False,desired,75)
                            label=float(np.clip((desired-actual)*1.5,-1,1))
                            item={'map':m,'seed':seed,'step':step,'base_speed_mps':base,
                                  'lane_class':'StraightLane','teacher_steer_md':0.0,
                                  'teacher_longitudinal':label,'native_idm_longitudinal':label,
                                  'applied_longitudinal':label,'longitudinal_clipped':0}
                            item.update(row);w.writerow(item)
                            counts[split]+=1
        assert counts[split]==576
    manifest={'status':'V36_NATIVE_IDM_LONGITUDINAL_TEACHER_DATA',
              'maps':['S','C','X'],'speeds_mps':[2.,3.,4.],
              'splits':groups,'data_fields':fields,
              'speed_schedule_enabled':True,'rows_by_split':counts}
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return counts


def test_fitting_and_leakage():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);fake_dataset(root)
        bundle=trainer.fit(root,PDD,max_iter=45)
        assert bundle['version']==VERSION
        val=bundle['metrics']['val']['overall']
        assert val['brake_n']>10 and val['throttle_n']>10
        assert math.isfinite(val['mae'])
        assert bundle['metrics']['val']['clip_fraction']==0
        x,y,m,s,t,c=trainer.read_rows(root/'train.csv',names(PDD))
        assert len(x)==576
        with (root/'train.csv').open('a',newline='',encoding='utf-8') as fp:
            writer=csv.DictWriter(fp,fieldnames=collector.CSV_META+names(PDD))
            rec={k:0 for k in collector.CSV_META+names(PDD)}
            rec.update(seed=76,map='C',base_speed_mps=2,teacher_longitudinal=.5,applied_longitudinal=.5,step=300,
                       target_speed_mps=2,speed_mps=2,remaining_route_m=20)
            writer.writerow(rec)
        raises(ValueError,trainer.fit,root,PDD,45)


def test_eval_patching_restored():
    feat={k:0. for k in PDD};feat.update(speed_mps=2.5)
    geom={'lateral_right_m':0.1,'heading_error_rad':0.02}
    cache={'feature_calls':0}
    class Simple:
        def predict(self,x):return [-.37]
    bundle={'model':Simple(),'feature_order':names(PDD)}
    real_f=evaluator.v30.feature_for_model
    real_g=evaluator.v30.planned_geometry
    real_l=evaluator.v30.governed_speed_action
    def observed(env,gap):return feat,{'remaining_m':100.,'adapted':False}
    def observed_geom(car,gap):return geom
    def governor(s,t,r):return .21,2.4
    evaluator.v30.feature_for_model=observed
    evaluator.v30.planned_geometry=observed_geom
    evaluator.v30.governed_speed_action=governor
    try:
        entries=[]
        with evaluator.longitudinal_control('learned_long_v36',bundle,PDD,entries):
            evaluator.v30.feature_for_model(None,2)
            evaluator.v30.planned_geometry(None,2)
            action,target=evaluator.v30.governed_speed_action(2.5,3,100)
            assert action==-.37 and target==2.4
            assert entries[0]['governor_action']==.21
        assert evaluator.v30.feature_for_model is observed
        assert evaluator.v30.planned_geometry is observed_geom
        assert evaluator.v30.governed_speed_action is governor
        entries=[]
        with evaluator.longitudinal_control('governor_v35',bundle,PDD,entries):
            assert evaluator.v30.governed_speed_action(2.5,3,100)==(.21,2.4)
    finally:
        evaluator.v30.feature_for_model=real_f
        evaluator.v30.planned_geometry=real_g
        evaluator.v30.governed_speed_action=real_l


def main():
    tests=[test_contracts,test_recorded_policy_contract,
           test_fitting_and_leakage,test_eval_patching_restored]
    for fn in tests:
        fn();print('PASS',fn.__name__)
    print('V3.6 OFFLINE TESTS PASS (does not validate live MetaDrive).')

if __name__=='__main__':main()
