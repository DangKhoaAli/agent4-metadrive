"""No MetaDrive import: synthetic geometry/API contract checks."""
from route_lookahead_v23 import road_center_point, collect_forward_segments, sample_route_waypoints

class Lane:
    def __init__(self,start,end,width=3.5):
        self.start=start;self.end=end;self.width=width;self.length=end[0]-start[0]
    def position(self,s,y):
        return (self.start[0]+s,self.start[1]-y)
    def local_coordinates(self, pos):
        return (pos[0]-self.start[0],self.start[1]-pos[1])

class Data: pass

def make_env(x=5.0, n_roads=3):
    graph={}
    for i,(a,b,begin,end) in enumerate([('A','B',0,10),('B','C',10,50),('C','D',50,100)][:n_roads]):
        graph.setdefault(a,{})[b]=[Lane((begin,7.0),(end,7.0)),
                                     Lane((begin,3.5),(end,3.5)),
                                     Lane((begin,0.0),(end,0.0))]
    env=Data();env.agent=Data();env.agent.position=(x,7.0)
    env.agent.convert_to_local_coordinates=lambda delta,_:delta
    env.agent.navigation=Data();nav=env.agent.navigation
    nav.checkpoints=['A','B','C','D'][:n_roads+1]
    nav._target_checkpoints_index=[0,1]
    nav.map=Data();nav.map.road_network=Data();nav.map.road_network.graph=graph
    return env

if __name__=='__main__':
    env=make_env()
    p,d=sample_route_waypoints(env,(25,45),-1)
    assert abs(p[0][0]-25)<1e-6 and abs(p[0][1]-3.5)<1e-6,p
    assert abs(p[1][0]-45)<1e-6 and abs(p[1][1]-3.5)<1e-6,p
    assert d['complete'] and d['geometry_continuous'],d
    env=make_env(x=9)
    p,d=sample_route_waypoints(env,(25,45),-1)
    assert abs(p[0][0]-25)<1e-6 and abs(p[1][0]-45)<1e-6, p
    env=make_env(n_roads=2)
    p,d=sample_route_waypoints(env,(25,55),-1)
    assert d['route_sample_statuses'][0]['complete']
    assert not d['route_sample_statuses'][1]['complete']
    assert not d['usable_for_shadow']
    try:
        sample_route_waypoints(make_env(),(45,25),-1)
        raise AssertionError('Expected invalid horizon error')
    except ValueError:
        pass
    print('Route lookahead V2.3 synthetic geometry tests: PASS')
