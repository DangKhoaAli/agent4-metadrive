"""Unit tests for guarded speed and steering conversion, no MetaDrive needed."""
import math
from guarded_pilot_v27 import adaptive_horizons,bounded_steering,governed_speed_action

for remaining in (1,5,15,25,44.999):
    (p,q),shortened=adaptive_horizons(remaining)
    assert shortened and 0<p<q<remaining, (remaining,p,q)
assert adaptive_horizons(45)[1] is True
assert adaptive_horizons(45.01)==((25.0,45.0),False)
assert adaptive_horizons(200)==((25.0,45.0),False)
for remaining in (0,-1,float('nan')):
    try:adaptive_horizons(remaining)
    except ValueError:pass
    else:raise AssertionError('Should reject remaining %s'%remaining)
assert bounded_steering(+0.2)==-0.2
assert bounded_steering(-0.2)==+0.2
assert bounded_steering(0.7)==-0.35
assert bounded_steering(0.5,previous=0.0,max_delta=0.1)==-0.1
assert bounded_steering(0.0,previous=0.2,max_delta=0.1)==0.1
assert governed_speed_action(0,3,100)[0]>0
assert governed_speed_action(10,3,100)[0]<0
assert governed_speed_action(3,3,10)[1]<3
assert all(math.isfinite(x) for x in governed_speed_action(3,3,10))
print('V2.7 bounded action and adaptive-horizon unit tests: PASS')
