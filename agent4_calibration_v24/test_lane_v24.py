"""Synthetic geometry tests; can run without MetaDrive."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent / "agent4_calibration_v23"))
from lane_route_v24 import choose_lane_chain, sample_lane_chain, sample_lane_route_waypoints

class StraightLane:
    def __init__(self,y,length=10.0):self.y,self.length,self.width=float(y),float(length),3.5
    def local_coordinates(self,pos):return float(pos[0]),float(pos[1])-self.y
    def position(self,lon,lat):return (float(lon),self.y+float(lat))

class ShiftedLane(StraightLane):
    def __init__(self,y,x0,length):super().__init__(y,length);self.x0=x0
    def local_coordinates(self,pos):return pos[0]-self.x0,pos[1]-self.y
    def position(self,lon,lat):return (self.x0+lon,self.y+lat)

class FakeAgent:
    position=(5.,7.)
    def convert_to_local_coordinates(self,delta,_):return delta

class Nav:
    checkpoints=['a','b','c']
    _target_checkpoints_index=[0,1]
    class M:pass
    map=M()
    map.road_network=M()
    map.road_network.graph={
        'a':{'b':[ShiftedLane(i*3.5,0.,10.) for i in range(3)]},
        'b':{'c':[ShiftedLane(i*3.5,10.,40.) for i in range(3)]}
    }

class Env:
    class Agent(FakeAgent):
        navigation=Nav()
    agent=Agent()


def test():
    env=Env()
    from route_lookahead_v23 import collect_forward_segments
    seg=collect_forward_segments(env.agent.navigation,env.agent)
    chain,gaps=choose_lane_chain(seg,env.agent.position)
    assert not gaps, gaps
    assert [x['lane_index'] for x in chain]==[2,2]
    p1,i1=sample_lane_chain(chain,25.)
    p2,i2=sample_lane_chain(chain,45.)
    assert p1==(30.,7.),p1
    assert p2==(50.,7.),p2
    assert i1['complete'] and i2['complete']
    pts,diag=sample_lane_route_waypoints(env,(25.,45.),-1)
    assert diag['usable_for_shadow']
    assert pts==[(25.,-0.),(45.,-0.)],pts
    # If a successor lane jumps laterally 7m, mark geometry invalid.
    seg[1]['lanes']=[ShiftedLane(i*3.5+20.,10.,40.) for i in range(3)]
    _,gaps=choose_lane_chain(seg,env.agent.position)
    assert gaps, 'Discontinuous successor must be rejected'
    # Insufficient route length is flagged (rather than silently extrapolated).
    _,status=sample_lane_chain(chain,200.)
    assert not status['complete']
    print('Lane-aware lookahead V2.4 synthetic tests: PASS')

if __name__=='__main__':test()
