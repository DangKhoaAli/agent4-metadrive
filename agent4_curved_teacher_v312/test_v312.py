"""No MetaDrive needed: fake-environment tests for V3.12 logic and provenance."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import curve_teacher_v312 as v
from geometry_v312 import lane_pose, compare_vehicle_route


class StraightLane:
    length=100.
    width=3.5
    index=('A','B',0)
    def local_coordinates(self,point):
        return float(point[0]),float(point[1])
    def position(self,s,lateral):
        return (s,lateral)

class FakeNav:
    def __init__(self,lane):
        self.current_ref_lanes=[lane]
        self.next_ref_lanes=[]

class FakeAgent:
    WIDTH=1.7
    LENGTH=4.0
    dist_to_left_side=1.75
    dist_to_right_side=1.75
    def __init__(self):
        self.lane=StraightLane()
        self.navigation=FakeNav(self.lane)
        self.position=[5.,.05]
        self.heading_theta=0.
        self.speed_km_h=10.8
        self.id='ego_0'
        self.name='ego_0'

class FakePolicy:
    def __init__(self):
        self.record_count=0
        self.last_recorded_action=None

class FakeEngine:
    def __init__(self,policy):self.policy=policy
    def get_policy(self,k):return self.policy

class FakeEnv:
    def __init__(self,finish_after=3,skip_action=False):
        self.agent=FakeAgent()
        self.engine=FakeEngine(FakePolicy())
        self.n=0
        self.finish_after=finish_after
        self.skip_action=skip_action
    def reset(self,seed=0):
        self.agent=FakeAgent()
        self.engine=FakeEngine(FakePolicy())
        self.n=0
    def step(self,action):
        assert action==[0.,0.], 'external action must not steer'
        if not self.skip_action:
            p=self.engine.policy
            p.last_recorded_action=[.1,.2]
            p.record_count+=1
        self.n+=1
        self.agent.position[0]+=.3
        done=self.n>=self.finish_after
        return None,0,done,False,{'arrive_dest':done,'route_completion':self.n/self.finish_after}

class TestV312(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(v.seed_list('158,160-162'),[158,160,161,162])
        for x in ('','1,1','1-0','1,x','-1'):
            with self.assertRaises(ValueError):v.seed_list(x)

    def test_geometry_sign(self):
        e=lane_pose(StraightLane(),[4.,.15],0.)
        # left of +x travel should be negative rightward lateral.
        self.assertAlmostEqual(e['lateral_right_m'],-.15)
        self.assertAlmostEqual(e['heading_error_rad'],0.)
        self.assertAlmostEqual(e['lane_curvature_proxy_rad_per_m'],0.)
        agent=FakeAgent()
        info=compare_vehicle_route(agent)
        self.assertFalse(info['vehicle_lane_differs_from_plan'])
        self.assertTrue(info['route_geometry_available'])
        self.assertEqual(info['planned']['lane_class'],'StraightLane')

    def test_attribution_good(self):
        env=FakeEnv(finish_after=3)
        ep=v.evaluate_episode(env,FakePolicy,'C',158,100,1,10,.35)
        self.assertEqual(ep['status'],'arrive_dest')
        self.assertTrue(ep['summary']['confirmed_success'])
        self.assertEqual(ep['summary']['steps'],3)
        self.assertEqual(ep['summary']['policy_action_count'],3)
        self.assertEqual(len(ep['trace']),3)
        self.assertEqual(ep['trace'][0]['idm_action'],[.1,.2])
        self.assertAlmostEqual(ep['summary']['mean_speed_mps'],3.)

    def test_attribution_must_fail_closed(self):
        env=FakeEnv(finish_after=3,skip_action=True)
        ep=v.evaluate_episode(env,FakePolicy,'C',158,100,1,10,.35)
        self.assertEqual(ep['status'],'episode_error')
        self.assertFalse(ep['summary']['confirmed_success'])
        self.assertEqual(ep['summary']['policy_capture_errors'],1)
        self.assertIn('IDM control attribution failed',ep['errors'][0])

    def test_event_threshold_and_map_transition(self):
        prev={'planned_lateral_right_m':.12,'geometry':{'vehicle_lane_id':'A','planned_lane_id':'A'}}
        curr={'planned_lateral_right_m':.35,'distance_to_lane_end_m':8,
              'geometry':{'vehicle_lane_id':'B','planned_lane_id':'B'}}
        marks=v.event_marks(curr,prev)
        self.assertIn('lateral_crossed_0.15m',marks)
        self.assertIn('lateral_crossed_0.30m',marks)
        self.assertIn('physical_lane_transition',marks)

    def test_prior_no_false_speed_matching(self):
        ep={'map':'C','seed':158,'profile':'idm_reference','status':'arrive_dest',
            'summary':{'confirmed_success':True,'mean_speed_mps':2.8}}
        ref={'episodes':[{'map':'C','seed':158,'profile':'learned_v310','speed_mps':3,
                          'summary':{'status':'out_of_road','confirmed_success':False}}]}
        rows=v.compare_prior([ep],ref)
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['speed_matched_verified'])

    def test_atomic_resume_and_csv(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            cfg={'maps':['C'],'seeds':[158],'idm_speed_kmh':10.8}
            report=v.report_template(cfg)
            ep=v.evaluate_episode(FakeEnv(),FakePolicy,'C',158,100,1,10,.35)
            report['episodes']=[ep]
            v.refresh(report)
            v.atomic_write(d/'result.json',report)
            v.export_csv(d/'result.csv',report)
            restored=json.loads((d/'result.json').read_text(encoding='utf8'))
            self.assertEqual(restored['group_scores'][0]['confirmed_success'],1)
            self.assertIn('confirmed_success',(d/'result.csv').read_text(encoding='utf-8-sig'))


if __name__=='__main__':
    unittest.main(verbosity=2)
