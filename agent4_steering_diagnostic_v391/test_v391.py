"""Lightweight offline verification; MetaDrive integration requires local machine."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_v391 import speeds_csv,speeds_stats,score,PROFILES

def test_parse():
    assert speeds_csv('4,6,8')==[4.,6.,8.]
    for val in ['','4,4','NaN','12.01','-2','a']:
        try:speeds_csv(val)
        except ValueError:pass
        else:raise AssertionError(val)

def test_capture_math():
    obs=[dict(speed_mps=2.,target_mps=4.),dict(speed_mps=3.,target_mps=4.)]
    r=speeds_stats(obs,2)
    assert r['max_speed_mps']==3. and r['mean_speed_mps']==2.5
    assert r['mean_abs_speed_error_mps']==1.5
    assert speeds_stats([],0)['max_speed_mps'] is None
    try:speeds_stats(obs,1)
    except RuntimeError:pass
    else:raise AssertionError('Failed to protect capture alignment')

def test_score():
    eps=[dict(map='S',speed_mps=4.,profile='learned_v35',seed=s,
              summary={'status':st}) for s,st in [(122,'arrive_dest'),(123,'lane_departure_guard')]]
    got=score(eps)
    assert len(got)==1 and got[0]['success']==1 and got[0]['total']==2
    assert got[0]['failures'][0]['seed']==123
    assert PROFILES['pdd_only']=='baseline_3'

if __name__=='__main__':
    test_parse();test_capture_math();test_score()
    print('V3.9.1 offline tests: PASS')
