"""Offline tests for V3.11B orchestration. No MetaDrive driving is performed."""
from __future__ import annotations
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
import sys
import types
from unittest.mock import patch
from pathlib import Path
import tempfile

from curve_eval_v311b import (VERSION,valid_speeds,validate_args,default_report,init_report,
                              enrich_episode,episode_is_success,aggregate,pairwise,
                              atomic_json,export_csv,keys,run)


def assert_reject(f, exc=(ValueError,FileExistsError)):
    try:f()
    except exc:return
    raise AssertionError('Expected input rejection')


def fixture(profile, status='arrive_dest', speed=2., lateral=.1, margin=.5, calls=5):
    sm=dict(status=status, n_applied_steps=calls, time_to_goal_s=50. if status=='arrive_dest' else None,
            max_abs_planned_lateral_right_m=lateral,
            min_estimated_edge_clearance_m=margin, near_edge_warning_steps=1,
            error_count=0, student_prediction_calls=calls,
            student_predictions_aligned=True, learned_steering_applied=profile=='learned_v310',
            measured_steps_aligned=True, teacher_fallback_calls=0)
    # Ground the synthetic learner in an explicit applied-action attribution.
    # V3.10 does NOT export measured_steps_aligned; fixture omits it on purpose.
    sm.pop('measured_steps_aligned', None)
    trace = ([{'step': 0, 'v310_attribution': {
                'pdd_base': .01, 'model_residual': .005,
                'model_proposed_action': .015, 'actual_applied_action': .015}}]
             if profile == 'learned_v310' else [])
    return enrich_episode(dict(map='C',seed=158,speed_mps=speed,profile=profile,
                               summary=sm,errors=[],trace=trace))


def test_parsing():
    assert valid_speeds('2,3,4')==[2.,3.,4.]
    for spec in ('', '2,2', 'nan', 'inf', '5', '0.5', 'x'):
        assert_reject(lambda spec=spec:valid_speeds(spec))
    args=Namespace(maps='C,X',seeds='158-159',speeds='2,3,4',
                   profiles='pdd_only,rule_lateral,learned_v310',steps=2500,
                   trace_stride=25,resume=False,overwrite=False)
    parsed=validate_args(args)
    assert parsed[0]==['C','X'] and parsed[1]==[158,159]
    args.seeds='157';assert_reject(lambda:validate_args(args))
    args.seeds='158';args.maps='Z';assert_reject(lambda:validate_args(args))
    args.maps='C';args.resume=True;args.overwrite=True
    assert_reject(lambda:validate_args(args))


def test_scoring():
    learner=fixture('learned_v310')
    teacher=fixture('rule_lateral',lateral=.11)
    pdd=fixture('pdd_only',status='out_of_road',lateral=.4)
    assert episode_is_success(learner) and episode_is_success(teacher)
    assert not episode_is_success(pdd)
    assert learner['summary']['out_of_v310_training_speed']
    learner_bad=fixture('learned_v310')
    learner_bad['summary']['student_prediction_calls']=4
    enrich_episode(learner_bad)
    assert not episode_is_success(learner_bad)
    learner_no_attribution=fixture('learned_v310')
    learner_no_attribution['trace'][0]['v310_attribution']['actual_applied_action'] = .04
    enrich_episode(learner_no_attribution)
    assert not episode_is_success(learner_no_attribution)
    explicit_bad=fixture('learned_v310')
    explicit_bad['summary']['measured_steps_aligned']=False
    enrich_episode(explicit_bad)
    assert not episode_is_success(explicit_bad)
    g=aggregate([learner,teacher,pdd])
    assert len(g)==3
    assert any(x['profile']=='learned_v310' and x['success']==1 for x in g)
    pairs=pairwise([learner,teacher,pdd])
    assert len(pairs)==2
    assert any(x['baseline']=='rule_lateral' and abs(x['learner_minus_baseline_max_lateral_m']+.01)<1e-9 for x in pairs)
    assert len({keys(x) for x in [learner,teacher,pdd]})==3


def test_resume_and_csv():
    with tempfile.TemporaryDirectory() as temp:
        dest=Path(temp)/'result.json'
        cfg={'maps':['C'],'seeds':[158],'speeds_mps':[2.], 'profiles':['learned_v310'],
             'steps':2500,'trace_stride':25,'pdd_sha256':'a','steering_sha256':'b'}
        r=init_report(dest,cfg,False,False)
        assert r['version']==VERSION
        r['episodes'].append(fixture('learned_v310'))
        atomic_json(dest,r)
        export_csv(r,dest.with_suffix('.csv'))
        old=init_report(dest,cfg,True,False)
        assert len(old['episodes'])==1
        assert 'confirmed_success' in old['episodes'][0]['summary']
        assert 'arrive_dest' in dest.with_suffix('.csv').read_text('utf-8-sig')
        assert_reject(lambda:init_report(dest,{**cfg,'steering_sha256':'x'},True,False))
        assert_reject(lambda:init_report(dest,cfg,False,False),exc=(FileExistsError,))
        assert init_report(dest,cfg,False,True)['episodes']==[]


def test_mocked_runner_checkpoint_resume():
    import curve_eval_v311b as v
    class FakeEnv:
        def __init__(self,opts):self.opts=opts;self.closed=False
        def close(self):self.closed=True
    meta=types.ModuleType('metadrive')
    envs=types.ModuleType('metadrive.envs')
    leaf=types.ModuleType('metadrive.envs.metadrive_env')
    leaf.MetaDriveEnv=FakeEnv
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp)
        pdd=temp/'pdd.joblib';pdd.write_bytes(b'fixture-pdd')
        steering=temp/'steering.joblib';steering.write_bytes(b'fixture-steering')
        out=temp/'curves.json'
        args=Namespace(pdd=pdd,steering=steering,maps='C,X',seeds='158',speeds='2',
                       profiles='pdd_only,learned_v310',steps=2500,trace_stride=25,
                       out=out,resume=False,overwrite=False)
        calls=[]
        def simulate(env,m,seed,speed,profile,pdd,model,names,steps,stride):
            calls.append((m,seed,speed,profile))
            e=fixture(profile,speed=speed)
            e.update(map=m,seed=seed)
            return e
        with patch.dict(sys.modules,{'metadrive':meta,'metadrive.envs':envs,
                                     'metadrive.envs.metadrive_env':leaf}):
            with patch.object(v.joblib,'load',return_value={'fixture':True}), \
                 patch.object(v,'check_pdd',return_value=['x']), \
                 patch.object(v,'check_bundle',return_value=True), \
                 patch.object(v.v310,'run_one',side_effect=simulate):
                with redirect_stdout(io.StringIO()):
                    report=run(args)
                    assert len(report['episodes'])==4
                    args.resume=True
                    again=run(args)
                assert len(again['episodes'])==4 and len(calls)==4
                assert out.is_file() and out.with_suffix('.csv').is_file()
                assert all(x['summary']['model_predictions_valid'] for x in again['episodes'])


if __name__=='__main__':
    test_parsing();test_scoring();test_resume_and_csv();test_mocked_runner_checkpoint_resume()
    print('Agent 4 V3.11B offline tests: PASS (orchestration only, NOT MetaDrive)')
