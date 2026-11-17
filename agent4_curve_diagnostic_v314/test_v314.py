"""Unit tests: offline, standard-library only."""
import tempfile
from pathlib import Path
import unittest
from analyze_v314 import review_episode, trace_findings, analyze, emit


class DiagnosticTests(unittest.TestCase):
    def example(self,**kwargs):
        e=dict(map='C',seed=190,speed_mps=3.,status='out_of_road',confirmed_success=False,
               quality_pass=False,n_applied_steps=30,n_verified_prediction_steps=30,
               model_predictions_valid=True,teacher_fallback_calls=0,idm_policy_used=False,
               max_abs_planned_lateral_m=.78,max_abs_heading_error_rad=.02,
               min_edge_clearance_proxy_m=.003,near_edge_warning_steps=7,
               has_circular_lane=True,trace=[])
        e.update(kwargs)
        return e
    def test_departure(self):
        tags=review_episode(self.example())['suspected_findings']
        self.assertIn('road_departure',tags)
        self.assertIn('near_edge_or_low_clearance',tags)
    def test_success_no_quality(self):
        e=self.example(status='arrive_dest',confirmed_success=True,max_abs_planned_lateral_m=1.74,
                       min_edge_clearance_proxy_m=.82,near_edge_warning_steps=0)
        tags=review_episode(e)['suspected_findings']
        self.assertIn('lateral_vs_clearance_proxy_disagreement',tags)
        self.assertIn('arrived_but_quality_failed',tags)
    def test_trace_discontinuity(self):
        trace=[dict(step=10,lateral_m=1.5),dict(step=20,lateral_m=-1.7)]
        self.assertEqual(len(trace_findings(trace)),1)
    def test_trace_normal(self):
        self.assertEqual(trace_findings([dict(step=0,lateral_m=.01),dict(step=20,lateral_m=.3)]),[])
    def test_predictions_provenance(self):
        tags=review_episode(self.example(teacher_fallback_calls=1))['suspected_findings']
        self.assertIn('model_provenance_issue',tags)
    def test_nonfinite(self):
        with self.assertRaises(ValueError):review_episode(self.example(max_abs_planned_lateral_m=float('nan')))
    def test_duplicate_rejection(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.json'
            p.write_text(json.dumps(dict(version='AGENT4_V313_IDM_IMITATION_STEERING',
                config={},episodes=[self.example(),self.example()])))
            with self.assertRaises(ValueError):analyze(p)
    def test_detailed_evaluator_trace_without_idm(self):
        import unittest.mock as mock
        import eval_diagnostic_v314 as ev
        from types import SimpleNamespace
        class FakeEnv:
            def __init__(self):
                self.agent=SimpleNamespace(current_action=[0,0])
                self.count=0
            def reset(self, seed):
                self.count=0
            def step(self, commands):
                self.agent.current_action=list(commands)
                self.count+=1
                return None,0.0,self.count>=2,False,{'arrive_dest':self.count>=2,'route_completion':1.0}
        pre=dict(planned_lateral_right_m=0.,heading_error_rad=0.,speed_mps=2.0,
                 edge_clearance_proxy_m=.8,is_circular_lane=1,curvature_proxy_rad_per_m=.05,
                 lookahead_5m_available=1,lookahead_10m_available=1,
                 lookahead_heading_5m_rad=.1,lookahead_heading_10m_rad=.2,
                 planned_lane_id='planned123',vehicle_lane_id='physical123',
                 planned_lane_class='CircularLane',vehicle_lane_class='CircularLane',
                 distance_to_lane_end_m=11.0,selected_next_lane_id='next1',
                 selected_next_lane_seam_m=.1,left_side_m=2.,right_side_m=2.)
        fake=FakeEnv()
        with mock.patch.object(ev,'state_before_action',return_value=pre), \
             mock.patch.object(ev,'steering_prediction',return_value=.12):
            ep=ev.run_one(fake,{},'C',186,2.0,steps=3,trace_stride=1)
        self.assertEqual(ep['status'],'arrive_dest')
        self.assertTrue(ep['quality_pass'])
        self.assertEqual(ep['n_verified_prediction_steps'],2)
        self.assertEqual(len(ep['trace']),2)
        self.assertEqual(ep['trace'][0]['planned_lane_id'],'planned123')
        self.assertEqual(ep['trace'][0]['vehicle_lane_id'],'physical123')
        self.assertEqual(ep['trace'][0]['predicted_steer'],.12)

    def test_emit(self):
        with tempfile.TemporaryDirectory() as d:
            import json
            f=Path(d)/'summary.json'
            e=review_episode(self.example())
            emit(dict(episodes=[e],summary={}),f)
            self.assertTrue(f.exists())
            self.assertTrue(f.with_suffix('.csv').exists())

if __name__=='__main__':unittest.main()
