"""Offline unit tests. No MetaDrive installation required for these checks."""
from __future__ import annotations
import csv,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common_v38 import (SCENARIOS,VERSION,STATUS,feature_names,front_features,
                        expert_follow,validate_splits,predict)
from traffic_v38 import phase_speed,resolve_lead_spawn_slot,scheduled_lead_longitudinal
from train_v38 import train


def run():
    # Regression: MetaDrive FirstPGBlock is ~10 m; requested lead gap
    # frequently falls on a following road segment (never the initial lane).
    class FakeLane:
        def __init__(self,length):self.length=length
        def local_coordinates(self,pos):return (pos[0],0.)
    current=FakeLane(10.)
    graph={'>':{'>>':[FakeLane(10.) for _ in range(3)]},
           '>>':{'>>>':[FakeLane(50.) for _ in range(3)]},
           '>>>':{'goal':[FakeLane(55.) for _ in range(3)]}}
    nav=SimpleNamespace(checkpoints=['>','>>','>>>','goal'])
    ego=SimpleNamespace(lane=current,lane_index=('>','>>',2),
                        position=(5.,0.),navigation=nav)
    assert resolve_lead_spawn_slot(ego,SimpleNamespace(graph=graph),24.) == (('>>','>>>',2),19.)
    assert resolve_lead_spawn_slot(ego,SimpleNamespace(graph=graph),12.) == (('>>','>>>',2),7.)
    assert resolve_lead_spawn_slot(ego,SimpleNamespace(graph=graph),35.) == (('>>','>>>',2),30.)
    try:
        resolve_lead_spawn_slot(ego,SimpleNamespace(graph={'>':graph['>']}),24.)
    except RuntimeError: pass
    else: raise AssertionError('Missing navigation lane incorrectly accepted')
    try:
        resolve_lead_spawn_slot(ego,SimpleNamespace(graph=graph),150.)
    except RuntimeError: pass
    else: raise AssertionError('Beyond-route spawn incorrectly accepted')
    assert scheduled_lead_longitudinal(0.,0.)<0
    assert scheduled_lead_longitudinal(3.,0.)<=-.5
    assert scheduled_lead_longitudinal(0.,2.)==.4
    assert abs(scheduled_lead_longitudinal(1.,1.))<1e-10
    assert scheduled_lead_longitudinal(3.,1.)<0
    assert phase_speed('stop_release',0)==0
    assert phase_speed('stop_release',120)==5
    assert phase_speed('brake_release',55)==0
    assert phase_speed('slow_release',160)==1
    absent=front_features(dict(present=False))
    assert absent['front_present']==0 and absent['front_ttc_s']==30
    lead=dict(present=True,gap_m=9.,speed_mps=0.,closing_mps=3.)
    assert front_features(lead)['front_ttc_s']==3
    idle,_=expert_follow(3.,4.,dict(present=False))
    slowing,target=expert_follow(3.,4.,lead)
    assert slowing<idle and target<4 and slowing<0
    assert expert_follow(1.,3.,dict(present=True,gap_m=2.,closing_mps=1.))[0]<0
    validate_splits(dict(train=[100],val=[112],eval=[116]))
    try:validate_splits(dict(train=[100],val=[100],eval=[116]))
    except ValueError:pass
    else:raise AssertionError('Seed leakage accepted')

    pdd=['speed_mps']+[f'pdd_fake_{i}' for i in range(14)]
    features=feature_names(pdd)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        splits=dict(train=[100,101],val=[112,113],eval=[116,117])
        fieldnames=['map','seed','scenario','step','base_speed_mps','teacher_longitudinal',
                    'applied_longitudinal','teacher_target_mps','route_target_mps']+features
        manifest=dict(status=STATUS,version=VERSION,splits=splits,data_fields=fieldnames,
                      accepted_rows={'train':0,'val':0},scenarios=list(SCENARIOS),
                      speeds_mps=[2.,3.,4.])
        rng=np.random.RandomState(17)
        for split in ('train','val'):
            with (root/f'{split}.csv').open('w',newline='',encoding='utf8') as f:
                wr=csv.DictWriter(f,fieldnames=fieldnames);wr.writeheader()
                for scenario in SCENARIOS:
                    for j in range(200):
                        base=float([2.,3.,4.][j%3]);present=float(j%5!=0)
                        gap=float(2+j%22);speed=float(j%5)
                        lead={'present':bool(present),'gap_m':gap,
                              'speed_mps':1.,'closing_mps':max(0,speed-1.)}
                        action,target=expert_follow(speed,base,lead)
                        row={x:0. for x in pdd};row['speed_mps']=speed
                        row.update(planned_lateral_right_m=0.,heading_error_rad=0.,
                                   adaptive_lookahead=0.,target_speed_mps=base,
                                   remaining_route_m=70.,speed_error_mps=base-speed)
                        row.update(front_features(lead))
                        wr.writerow(dict(map='S',seed=splits[split][j%2],scenario=scenario,
                             step=j,base_speed_mps=base,teacher_longitudinal=action,
                             applied_longitudinal=action,teacher_target_mps=target,
                             route_target_mps=base,**row))
                        manifest['accepted_rows'][split]+=1
        (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
        model=train(root,pdd,iterations=55)
        pred=predict(model,{k:float(0) for k in features})
        assert -1<=pred<=1
        assert model['metrics']['val']['all']['n']==600
    print('Agent 4 V3.8 offline tests: PASS')
if __name__=='__main__':run()
