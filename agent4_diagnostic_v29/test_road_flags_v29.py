"""Unit-test exact out-of-road predicate mirrors MetaDrive 0.4.3."""
from diagnose_road_flags_v29 import out_of_road_causes

base={'on_lane':True,'out_of_route':False,'on_yellow_continuous_line':False,
      'on_white_continuous_line':False,'on_broken_line':False,'crash_sidewalk':False}
cfg={'out_of_route_done':False,'on_continuous_line_done':True,'on_broken_line_done':False}
assert out_of_road_causes(base,cfg) == ([],False)
f={**base,'on_white_continuous_line':True}
assert out_of_road_causes(f,cfg) == (['on_white_continuous_line'],True)
f={**base,'on_lane':False,'on_yellow_continuous_line':True}
assert out_of_road_causes(f,cfg) == (['not_on_lane','on_yellow_continuous_line'],True)
f={**base,'out_of_route':True,'on_yellow_continuous_line':True}
assert out_of_road_causes(f,{**cfg,'out_of_route_done':True}) == (['out_of_route'],True)
f={**base,'on_broken_line':True}
assert out_of_road_causes(f,{**cfg,'on_broken_line_done':True}) == (['on_broken_line'],True)
f={**base,'on_lane':None}
assert out_of_road_causes(f,cfg) == ([],None)
print('V2.9 road predicate tests: PASS')
