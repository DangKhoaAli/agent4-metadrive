"""Offline regression tests; no MetaDrive installation required."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import collect_v312b as c
from validate_v312b import check_dataset


class FakeLane:
    def __init__(self,start=0.0,length=10.0,index=('A','B',0),heading=0.0,y0=0.0):
        self.start=float(start);self.length=float(length);self.index=index;self.heading=heading;self.y0=y0
    def local_coordinates(self,point):
        return float(point[0])-self.start,float(point[1])-self.y0
    def position(self,s,lateral):
        return (self.start+s,self.y0+lateral)


class FakeNavigation:
    def __init__(self):
        self.lane=FakeLane(0,10,('A','B',0))
        self.upcoming=FakeLane(10,20,('B','C',0))
        self.current_ref_lanes=[self.lane];self.next_ref_lanes=[self.upcoming]


class FakeAgent:
    WIDTH=1.7;LENGTH=4.0
    def __init__(self):
        self.id='ego_0';self.name='ego_0';self.position=(2.0,0.01)
        self.heading_theta=0.0;self.speed_km_h=10.8
        self.steering=0.0;self.throttle_brake=0.0
        self.current_action=(0.0,0.0)
        self.dist_to_left_side=1.65;self.dist_to_right_side=1.65
        self.navigation=FakeNavigation();self.lane=self.navigation.lane


class FakePolicy:
    def __init__(self,large_action=False):
        self.record_count=0;self.last_recorded_action=None;self.large_action=large_action
    def act(self):
        raw=[2.7,0.0] if self.large_action and self.record_count==2 else [.09,.01]
        self.record_count+=1;self.last_recorded_action=raw
        return raw


class FakeEngine:
    def __init__(self,policy):self.policy=policy
    def get_policy(self,key):return self.policy


class FakeEnvironment:
    def __init__(self,config,policy_type=FakePolicy,fail=False,no_clip=False,overshoot=False):
        self.config=config;self.policy_type=policy_type
        self.fail=fail;self.no_clip=no_clip;self.overshoot=overshoot
        self.reset(seed=0)
    def reset(self,seed=0):
        self.agent=FakeAgent();self.engine=FakeEngine(self.policy_type(self.overshoot));self.step_count=0
        return None,{}
    def step(self,action):
        assert action==[0.,0.]
        raw=self.engine.policy.act()
        applied=raw if self.no_clip else [c.clip(x) for x in raw]
        self.agent.current_action=applied
        self.agent.steering=applied[0];self.agent.throttle_brake=applied[1]
        self.step_count+=1
        self.agent.position=(self.agent.position[0]+.3,self.agent.position[1])
        done=self.step_count>=4
        info={'arrive_dest':done and not self.fail,'out_of_road':done and self.fail,
              'route_completion':.98 if done else .2}
        return None,0.0,done,False,info
    def close(self):pass


def args(out,**kw):
    data=dict(maps='C,X',speeds='3',train_seeds='170',val_seeds='171',test_seeds='172',
              steps=15,edge_warning_m=.35,max_lateral_m=.6,max_heading_rad=.12,
              min_clearance_m=.3,min_episode_steps=2,out_dir=Path(out),resume=False)
    data.update(kw)
    return argparse.Namespace(**data)


def fake_factory(config):return FakeEnvironment(config)

def fake_policy_type(_speed_kmh):return FakePolicy


class CollectTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(c.parse_seed_spec('10,12-14'),[10,12,13,14])
        self.assertEqual(c.parse_maps('S,C,X'),['S','C','X'])
        self.assertEqual(c.parse_speeds('2,3,4'),[2.,3.,4.])
        for s in ('2,2','3-2','1,b'):
            with self.assertRaises(ValueError):c.parse_seed_spec(s)
        with self.assertRaises(ValueError):c.split_seed_lists(args('no',val_seeds='170'))
        with self.assertRaises(ValueError):c.parse_speeds('2,2')

    def test_policy_action_raw_applied_clipping(self):
        car=FakeAgent()
        car.current_action=[1.,-.2];car.steering=1.;car.throttle_brake=-.2
        self.assertEqual(c.read_applied_action(car,[2.7,-.2]),[1.,-.2])
        car.current_action=[2.7,-.2]
        with self.assertRaises(RuntimeError):c.read_applied_action(car,[2.7,-.2])
        car.current_action=[1.,-.2];car.steering=0.
        with self.assertRaises(RuntimeError):c.read_applied_action(car,[2.7,-.2])

    def test_lookahead_across_lane_boundary(self):
        car=FakeAgent()
        features=c.lookahead_features(car)
        self.assertEqual(features['lookahead_5m_available'],1)
        self.assertEqual(features['lookahead_10m_available'],1)
        self.assertEqual(features['lookahead_20m_available'],1)
        self.assertAlmostEqual(features['lookahead_heading_10m_rad'],0.)
        self.assertIn('B',features['selected_next_lane_id'])
        # Next road may list lane 2 after lane 0. Select by seam geometry, not order.
        far=FakeLane(10,20,('B','C',2),y0=3.5)
        car.navigation.next_ref_lanes=[far,car.navigation.upcoming]
        chosen=c.lookahead_features(car)
        self.assertEqual(chosen['selected_next_lane_id'],repr(car.navigation.upcoming.index))
        self.assertAlmostEqual(chosen['selected_next_lane_seam_m'],0.)
        car.navigation.next_ref_lanes=[far]
        self.assertEqual(c.lookahead_features(car)['lookahead_10m_available'],0)
        car.navigation.next_ref_lanes=[]
        features=c.lookahead_features(car)
        self.assertEqual(features['lookahead_10m_available'],0)
        self.assertIsNone(features['lookahead_curvature_10m'])

    def test_row_flags_raw_over_range(self):
        pre=c.state_before_action(FakeAgent())
        row=c.make_step_row(pre,[2.7,.01],[1.,.01],'train','C',170,3.,2,{},False,False)
        self.assertEqual(row['raw_out_of_range'],1)
        self.assertEqual(row['applied_steering'],1.0)
        self.assertEqual(row['trainable_row'],0)
        self.assertIn('raw_out_of_range',row['excluded_reason'])
        self.assertEqual(row['current_action_verified'],1)

    def test_end_to_end_resume_and_seed_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=args(Path(tmp)/'dataset')
            man=c.run(a,environment_factory=fake_factory,policy_class_factory=fake_policy_type)
            self.assertEqual(man['totals']['episodes'],6)
            self.assertEqual(man['totals']['success'],6)
            self.assertEqual(man['totals']['clean_steps'],24)
            self.assertEqual(check_dataset(Path(tmp)/'dataset')['clean_steps'],24)
            self.assertEqual(man['totals']['by_split']['train']['episodes'],2)
            train=Path(tmp)/'dataset'/'clean'/'train'/'C'/'seed170_speed3p0mps.csv'
            self.assertTrue(train.is_file())
            with train.open(encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[0]['split'],'train')
            self.assertEqual(rows[0]['current_action_verified'],'1')
            self.assertAlmostEqual(float(rows[0]['applied_steering']),.09)
            a.resume=True
            resumed=c.run(a,environment_factory=fake_factory,policy_class_factory=fake_policy_type)
            self.assertEqual(resumed['totals']['episodes'],6)
            a.speeds='3,4'
            with self.assertRaises(ValueError):c.run(a,environment_factory=fake_factory,policy_class_factory=fake_policy_type)

    def test_overrange_kept_audit_filtered_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            fac=lambda conf:FakeEnvironment(conf,overshoot=True)
            a=args(Path(tmp)/'ds',maps='C',train_seeds='170',val_seeds='',test_seeds='')
            man=c.run(a,environment_factory=fac,policy_class_factory=fake_policy_type)
            self.assertEqual(man['totals']['raw_overrange_steps'],1)
            self.assertEqual(man['totals']['audit_steps'],4)
            self.assertEqual(man['totals']['clean_steps'],3)
            self.assertEqual(check_dataset(Path(tmp)/'ds')['raw_overrange_steps'],1)
            raw=Path(tmp)/'ds'/'audit'/'train'/'C'/'seed170_speed3p0mps.csv'
            with raw.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
            self.assertEqual(rows[2]['applied_steering'],'1.0')
            self.assertEqual(rows[2]['raw_steering'],'2.7')
            self.assertEqual(rows[2]['trainable_row'],'0')

    def test_fail_quarantines_entire_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            fac=lambda conf:FakeEnvironment(conf,fail=True)
            a=args(Path(tmp)/'ds',maps='C',train_seeds='170',val_seeds='',test_seeds='')
            man=c.run(a,environment_factory=fac,policy_class_factory=fake_policy_type)
            self.assertEqual(man['totals']['success'],0)
            self.assertEqual(man['totals']['clean_steps'],0)
            self.assertEqual(man['totals']['audit_steps'],4)
            self.assertEqual(check_dataset(Path(tmp)/'ds')['clean_steps'],0)
            self.assertFalse(any((Path(tmp)/'ds'/'clean').rglob('*.csv')))

    def test_unverified_action_quarantines_and_blocks_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            fac=lambda conf:FakeEnvironment(conf,no_clip=True,overshoot=True)
            a=args(Path(tmp)/'ds',maps='C',train_seeds='170',val_seeds='',test_seeds='')
            man=c.run(a,environment_factory=fac,policy_class_factory=fake_policy_type)
            e=man['episodes'][0]
            self.assertEqual(e['status'],'episode_error')
            self.assertEqual(e['steps'],2)
            self.assertEqual(e['actual_trainable_steps'],0)
            self.assertFalse(e['recorded_actions_aligned'])
            self.assertTrue(e['errors'])


if __name__=='__main__': unittest.main(verbosity=2)
