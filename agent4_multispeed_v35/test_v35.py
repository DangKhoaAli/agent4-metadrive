"""Offline self-tests V3.5. No MetaDrive or real model required."""
from __future__ import annotations
import csv
import json
import math
import tempfile
from pathlib import Path
from collections import Counter
import numpy as np

from collect_multispeed_v35 import validate_groups,speed_folder,merge_collected
from train_multispeed_v35 import equal_speed_map_weights,train,read_target_speeds
from evaluate_multispeed_v35 import check_student
from common_v33 import feature_names


def must_fail(callback,exc_type=ValueError):
    try:callback()
    except exc_type:return
    raise AssertionError(f'Expected {exc_type}')


def synthetic_experiment():
    pdd_features=['speed_mps']+[f'feature_{i}' for i in range(14)]
    names=feature_names(pdd_features)
    source_fields=['map','seed','step','teacher_steer_md']+names
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        speeds=[2.,3.,4.];maps=['S','C','X']
        train_s=[40,41]; val_s=[52]; holdout=[56,57]
        for speed in speeds:
            folder=root/speed_folder(speed)
            folder.mkdir()
            episodes=[]
            for split,seeds,nrows in (('train',train_s,60),('val',val_s,60)):
                count=0
                with (folder/f'{split}.csv').open('w',encoding='utf-8',newline='') as fp:
                    wr=csv.DictWriter(fp,fieldnames=source_fields)
                    wr.writeheader()
                    for m in maps:
                        for seed in seeds:
                            for j in range(nrows):
                                x=float(j)/nrows
                                base=0.03*math.sin(3*x)
                                row={'map':m,'seed':seed,'step':j,
                                     'teacher_steer_md':base+0.015*(speed-3.)+0.004*x,
                                     **{f:0.0 for f in pdd_features},
                                     'base_pdd_steer_md':base,
                                     'planned_lateral_right_m':x-0.5,
                                     'heading_error_rad':x*0.04,
                                     'adaptive_lookahead':0.0}
                                row['speed_mps']=speed+0.005*x
                                wr.writerow(row)
                                count+=1
                            episodes.append({'map':m,'seed':seed,'split':split,
                                             'status':'arrive_dest','rows_accepted_for_training':nrows})
            report={'status':'V33_SLOW_IDM_TEACHER_DATA_NOT_LEARNED_AGENT',
                    'splits':{'train':train_s,'val':val_s,'eval':holdout},
                    'maps':maps,'speed_target_mps':speed,'perturbation':.008,
                    'rows_by_split':{'train':len(train_s)*len(maps)*60,
                                     'val':len(val_s)*len(maps)*60},
                    'episodes':episodes}
            (folder/'manifest.json').write_text(json.dumps(report),encoding='utf-8')
        manifest=merge_collected(root,speeds,train_s,val_s,holdout,maps,.008)
        assert manifest['rows_by_split']=={'train':1080,'val':540}
        assert set(read_target_speeds(root/'train.csv'))==set(speeds)
        result=train(root,pdd_features,max_iter=55)
        assert result['version']=='V35_MULTISPEED_IDM_DISTILLED_RESIDUAL'
        assert result['metrics']['val']['student_v35']['steering_mae']<0.03
        check_student(result,pdd_features,holdout,result['version'])
        must_fail(lambda:check_student(result,pdd_features,[40],result['version']))
        must_fail(lambda:check_student(result,pdd_features,[39],result['version']))


def main():
    validate_groups([40,41],[52,53],[56,57])
    must_fail(lambda:validate_groups([40],[40],[56]))
    must_fail(lambda:validate_groups([36],[52],[56]))
    groups=['C','C','X','X','C']
    speeds=np.array([2,2,2,4,4])
    weights=equal_speed_map_weights(speeds,groups)
    totals={g:float(sum(w for (s,m),w in zip(zip(speeds,groups),weights) if (float(s),m)==g)) for g in set(zip([float(s) for s in speeds],groups))}
    assert len(totals)==4
    assert max(totals.values())-min(totals.values())<1e-9
    synthetic_experiment()
    print('Agent 4 V3.5 synthetic dataset / speed balancing / model training / seed isolation: PASS')


if __name__=='__main__':main()
