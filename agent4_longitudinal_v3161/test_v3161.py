"""Offline V3.16.1 contracts. Does NOT certify closed-loop MetaDrive behavior."""
from __future__ import annotations
import math,sys,unittest,json,tempfile
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common_v3161 import FilterConfig, StabilizedLongitudinal, VERSION
from eval_v3161 import RULE, RAW, STABLE, summary_for, compare, load_reference
from analyze_v316 import analyze


def sample(profile='rule_governor_v313',seed=186,speed=2.,quality=True,mae=.20,brakes=10):
    return dict(profile=profile,map='C',seed=seed,speed_mps=speed,
        confirmed_success=True,quality_pass=quality,status='arrive_dest',
        steps=100,verified_steps=100,speed_mae_mps=mae,
        model_brake_steps=brakes,overspeed_steps=0,errors=[])


def triple(stable_quality=True,baseline_quality=True,raw_quality=True,stable_brakes=20,raw_brakes=40):
    return [sample(RULE,quality=baseline_quality,mae=.2,brakes=5),
        sample(RAW,quality=raw_quality,mae=.17,brakes=raw_brakes),
        sample(STABLE,quality=stable_quality,mae=.18,brakes=stable_brakes)]


class ConfigTests(unittest.TestCase):
    def test_valid_defaults(self):self.assertEqual(FilterConfig().max_throttle,.70)
    def test_valid_mapping(self):self.assertEqual(FilterConfig().to_dict()['deadband'],.065)
    def test_up_rate_zero_rejected(self):
        with self.assertRaises(ValueError):FilterConfig(up_rate=0)
    def test_down_rate_gt_one(self):
        with self.assertRaises(ValueError):FilterConfig(down_rate=2)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):FilterConfig(max_step_down=float('nan'))
    def test_inf_rejected(self):
        with self.assertRaises(ValueError):FilterConfig(deadband=float('inf'))
    def test_negative_cap_rejected(self):
        with self.assertRaises(ValueError):FilterConfig(max_brake=-1)
    def test_high_deadband_rejected(self):
        with self.assertRaises(ValueError):FilterConfig(deadband=.4)


class FilterTests(unittest.TestCase):
    def test_first_action_cap(self):self.assertEqual(StabilizedLongitudinal().apply(1),.70)
    def test_first_brake_cap(self):self.assertEqual(StabilizedLongitudinal().apply(-1),-.80)
    def test_deadband_small_positive(self):self.assertEqual(StabilizedLongitudinal().apply(.01),0)
    def test_deadband_small_negative(self):self.assertEqual(StabilizedLongitudinal().apply(-.03),0)
    def test_zero(self):self.assertEqual(StabilizedLongitudinal().apply(0),0)
    def test_nonfinite_rejected(self):
        for x in [float('nan'),float('inf'),-float('inf')]:
            with self.subTest(x=x),self.assertRaises(ValueError):StabilizedLongitudinal().apply(x)
    def test_invalid_model_bound_rejected(self):
        with self.assertRaises(ValueError):StabilizedLongitudinal().apply(-2)
    def test_up_slew(self):
        ctl=StabilizedLongitudinal();ctl.apply(0)
        self.assertLessEqual(ctl.apply(1),.16+1e-10)
    def test_down_slew(self):
        ctl=StabilizedLongitudinal();ctl.apply(.7)
        self.assertGreaterEqual(ctl.apply(-1),.7-.35-1e-10)
    def test_reset_between_episodes(self):
        ctl=StabilizedLongitudinal();ctl.apply(.7);ctl.apply(-.8);ctl.reset()
        self.assertEqual(ctl.apply(-.8),-.8)
    def test_is_deterministic(self):
        seq=[1,.7,.3,-.2,-.7,0,.02,0.8]
        a=StabilizedLongitudinal();b=StabilizedLongitudinal()
        self.assertEqual([a.apply(x) for x in seq],[b.apply(x) for x in seq])
    def test_stays_bounded_random_sequence(self):
        ctl=StabilizedLongitudinal()
        for i in range(1000):
            value=ctl.apply(math.sin(i*.15))
            self.assertLessEqual(value,.7+1e-10)
            self.assertGreaterEqual(value,-.8-1e-10)
    def test_stabilization_reduces_total_variation_on_jitter(self):
        jitter=[.1,-.1]*50
        ctl=StabilizedLongitudinal()
        output=[ctl.apply(x) for x in jitter]
        raw_tv=sum(abs(a-b) for a,b in zip(jitter,jitter[1:]))
        filter_tv=sum(abs(a-b) for a,b in zip(output,output[1:]))
        self.assertLess(filter_tv,raw_tv)


class ABEvaluationTests(unittest.TestCase):
    def test_accept_good_paired_triple(self):
        self.assertTrue(compare(triple(),'validation')['accepted'])
    def test_probe_not_full_suite(self):
        result=compare(triple(),'validation',require_full_suite=True)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['full_validation_suite'])
    def test_final_never_accepts(self):
        self.assertFalse(compare(triple(),'final')['accepted'])
    def test_reject_rule_quality_regression(self):
        self.assertFalse(compare(triple(stable_quality=False),'validation')['accepted'])
    def test_reject_raw_quality_regression(self):
        self.assertFalse(compare(triple(stable_quality=False,baseline_quality=False),'validation')['accepted'])
    def test_reject_missing_raw(self):
        rows=triple();rows.pop(1)
        self.assertFalse(compare(rows,'validation')['accepted'])
    def test_reject_no_learned_brakes(self):
        self.assertFalse(compare(triple(stable_brakes=0),'validation')['accepted'])
    def test_reject_no_brake_reduction(self):
        self.assertFalse(compare(triple(stable_brakes=40,raw_brakes=40),'validation')['accepted'])
    def test_reject_speed_regression(self):
        rows=triple();rows[2]['speed_mae_mps']=.9
        self.assertFalse(compare(rows,'validation')['accepted'])
    def test_reject_quality_duplicate(self):
        rows=triple();rows.append(deepcopy(rows[1]))
        with self.assertRaises(ValueError):compare(rows,'validation')
    def test_summary_reference_no_variation(self):
        result=summary_for(triple())
        self.assertIsNone(result[RAW]['action_total_variation'])
    def test_quality_improvement_accounted(self):
        result=compare(triple(baseline_quality=False,raw_quality=False),'validation')
        self.assertTrue(result['accepted'])


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path('/mnt/data/agent4_v316_validation.json')
        cls.reference=path if path.is_file() else None
    def test_real_validation_reference_subset(self):
        if self.reference is None:self.skipTest('Previously uploaded V3.16 JSON not mounted')
        j=json.loads(self.reference.read_text(encoding='utf8'))
        cfg=j['config']
        reused=load_reference(self.reference,cfg['model_sha256'],cfg['baseline_sha256'],
            ['C'],[186,189],[2.],2500)
        self.assertEqual(len(reused),4)
        self.assertTrue(all(e['profile'] in (RULE,RAW) for e in reused))
    def test_reference_wrong_model_fails(self):
        if self.reference is None:self.skipTest('Previously uploaded V3.16 JSON not mounted')
        j=json.loads(self.reference.read_text(encoding='utf8'))
        with self.assertRaises(ValueError):load_reference(self.reference,'wrong',
            j['config']['baseline_sha256'],['C'],[186],[2.],2500)
    def test_reference_wrong_scenario_fails(self):
        if self.reference is None:self.skipTest('Previously uploaded V3.16 JSON not mounted')
        j=json.loads(self.reference.read_text(encoding='utf8'))
        with self.assertRaises(ValueError):load_reference(self.reference,j['config']['model_sha256'],
            j['config']['baseline_sha256'],['C'],[202],[2.],2500)
    def test_reproduces_exact_prior_regressions(self):
        if self.reference is None:self.skipTest('Previously uploaded V3.16 JSON not mounted')
        j=json.loads(self.reference.read_text(encoding='utf8'))
        d=analyze(j)
        self.assertEqual(d['n_pairs'],36)
        self.assertEqual([(e['map'],e['seed'],e['speed_mps']) for e in d['regressions']],
            [('C',186,2.),('C',189,2.)])
        self.assertEqual(d['total_brakes_learned'],4292)

if __name__=='__main__':unittest.main(verbosity=2)
