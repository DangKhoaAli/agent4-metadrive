"""Unit tests do not need MetaDrive runtime."""
from __future__ import annotations
import math
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_features_v32 as module


def run():
    v30=module.v30
    oldfeat, oldgeo=v30.feature_for_model,v30.planned_geometry
    feature={k:0.1 * (i+1) for i,k in enumerate(module.FEATURES)}
    route={'remaining_m':180.0,'effective_lookahead_m':[25.0,45.0],
           'adapted':False,'lane_indices':[('a','b',0)]}
    g={'lane':'dummy','lane_index':('a','b',0),'current_vehicle_lane':"('a','b',0)",
       'tangent_rad':0.0,'local_lane_lateral_to_right_sign':-1}
    env=SimpleNamespace(agent=SimpleNamespace())
    actions=[]
    def mockfeat(env,max_gap):
        assert max_gap==2.0
        return feature, route
    def mockgeom(car,max_gap):
        assert car is env.agent
        return g
    def mockpilot(env,seed,model,mode,options):
        f,r=v30.feature_for_model(env,2.0)
        geom=v30.planned_geometry(env.agent,2.0)
        assert f is feature and r is route and geom is g
        actions.append((mode,seed))
        return {'n_applied_steps':1,'status':'step_limit','errors':[],
                'travelled_approx_m':0.3,
                'trace':[{'step':0,'model_steer_md':0.1,'residual_steer':-.04,
                          'planned_lateral_right_m':-.3,'heading_error_rad':.05}]}
    try:
        v30.feature_for_model=mockfeat; v30.planned_geometry=mockgeom
        output=module.capture_episode(env,2,{},'lateral_3',pilot=mockpilot,
                                      steps=10,trace_stride=1)
        assert v30.feature_for_model is mockfeat and v30.planned_geometry is mockgeom
        assert output['audit']['audit_alignment_ok']
        assert output['audit']['n_trace_points_with_features']==1
        assert output['trace'][0]['feature_audit']['features_pdd_v2']==feature
        assert output['trace'][0]['geometry_audit']['planned_lane_index']=="('a', 'b', 0)"
        assert actions==[('residual',2)]
        assert output['trace'][0]['model_and_residual_opposed'] is True
        raw = -.3*.08 - .05*.30
        assert math.isclose(output['trace'][0]['raw_residual_unclipped'],raw)
        ref={'episodes_by_map':[{'map':'C','episodes':[{'seed':2,'profile':'lateral_3',
             'summary':{'status':'step_limit','n_applied_steps':1,'travelled_approx_m':.3}}]}]}
        cp=module.reference_comparison(ref,'C',2,'lateral_3',output)
        assert cp['near_exact_reproduction'] is True
        assert module.reference_comparison(ref,'X',2,'lateral_3',output)['found'] is False
    finally:
        v30.feature_for_model=oldfeat; v30.planned_geometry=oldgeo
    print('Agent 4 V3.2 feature capture and no-policy-change unit tests: PASS')

if __name__=='__main__':
    run()
