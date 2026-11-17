"""Shadow-only, lane-consistent MetaDrive route target sampler (Agent4 V2.4).

No model actions are applied. This is an experimental alternative to the
road-WIDE centerline used by V2.3; it cannot guarantee planned lane changes.
"""
from __future__ import annotations
import math
from route_lookahead_v23 import collect_forward_segments


def _vec2(p):
    a,b=float(p[0]),float(p[1])
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError('Non-finite lane point')
    return (a,b)


def lane_center(lane, longitudinal):
    return _vec2(lane.position(float(longitudinal), 0.0))


def _d(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def choose_lane_chain(segments, ego_position, max_gap_m=2.0):
    """Choose an initial lane closest to ego, then successor by endpoint.

    This only preserves geometric continuity. It does NOT decide which lane the
    destination requires (e.g. a protected turn), so it is diagnostic only.
    """
    if not segments:
        raise ValueError('No route segments')
    ego=_vec2(ego_position)
    chosen=[]
    lane0_candidates=[]
    for idx, lane in enumerate(segments[0]['lanes']):
        lon, lat=map(float, lane.local_coordinates(ego))
        if not(math.isfinite(lon) and math.isfinite(lat)):
            continue
        length=float(lane.length)
        # Centerline proximity. Reject pathological projections outside road.
        offroad_long=max(0.0,-lon,lon-length)
        score=abs(lat)+3.0*offroad_long
        lane0_candidates.append((score,idx,lon))
    if not lane0_candidates:
        raise ValueError('Cannot project ego onto starting lanes')
    _,idx,lon=min(lane0_candidates)
    first=segments[0]
    start=max(0.0,min(lon,float(first['lanes'][idx].length)))
    chosen.append({'lanes':first['lanes'],'lane_index':idx,'lane':first['lanes'][idx],
                   'start_longitude':start,'nodes':first['nodes']})
    gaps=[]
    for seg in segments[1:]:
        prev=chosen[-1]
        p_end=lane_center(prev['lane'],float(prev['lane'].length))
        options=[]
        for candidate_idx,lane in enumerate(seg['lanes']):
            gap=_d(p_end,lane_center(lane,0.0))
            # Small index difference tie-breaker; gap is primary criterion.
            options.append((gap,abs(candidate_idx-prev['lane_index']),candidate_idx,lane))
        if not options:
            raise ValueError('Missing successor lanes')
        gap,_,i,next_lane=min(options)
        if gap>max_gap_m:
            gaps.append({'from':list(map(str,prev['nodes'])),
                         'to':list(map(str,seg['nodes'])),
                         'gap_m':round(gap,6),'previous_lane_index':prev['lane_index'],
                         'new_lane_index':i})
        chosen.append({'lanes':seg['lanes'],'lane_index':i,'lane':next_lane,
                       'start_longitude':0.0,'nodes':seg['nodes']})
    return chosen, gaps


def sample_lane_chain(chosen, distance_m):
    remaining=float(distance_m)
    if not math.isfinite(remaining) or remaining<0:
        raise ValueError('Lookahead must be nonnegative finite')
    travelled=0.0
    for seg in chosen:
        length=float(seg['lane'].length)
        start=seg['start_longitude']
        remaining_this=max(0.0,length-start)
        if remaining<=remaining_this+1.e-6:
            s=min(length,start+max(0.0,remaining))
            return lane_center(seg['lane'],s),{
                'requested_m':float(distance_m), 'achieved_m':travelled+s-start,
                'complete':True, 'lane_index':seg['lane_index'],
                'road':list(map(str,seg['nodes']))}
        remaining-=remaining_this
        travelled+=remaining_this
    final=chosen[-1]
    return lane_center(final['lane'],float(final['lane'].length)),{
        'requested_m':float(distance_m),'achieved_m':travelled,
        'complete':False,'lane_index':final['lane_index'],
        'road':list(map(str,final['nodes']))}


def sample_lane_route_waypoints(env, horizons=(25.0,45.0),
                                y_to_right_multiplier=-1, max_gap_m=2.0):
    if len(horizons)!=2 or not 0<horizons[0]<horizons[1]:
        raise ValueError('Expected increasing two-lookahead distances')
    if y_to_right_multiplier not in (-1,1):
        raise ValueError('Invalid axis multiplier')
    agent=env.agent
    segments=collect_forward_segments(agent.navigation, agent)
    chosen,gaps=choose_lane_chain(segments,agent.position,max_gap_m)
    wp,world,details=[],[],[]
    px,py=_vec2(agent.position)
    for horizon in horizons:
        p,info=sample_lane_chain(chosen,horizon)
        dx,dy=agent.convert_to_local_coordinates((p[0]-px,p[1]-py),0.0)
        x,y=float(dx),float(dy)*y_to_right_multiplier
        if not(math.isfinite(x) and math.isfinite(y)):
            raise ValueError('Invalid ego-frame projection')
        wp.append((x,y));world.append(p);details.append(info)
    good=all(i['complete'] for i in details) and len(gaps)==0
    return wp,{
        'route_waypoints_world_xy': [list(a) for a in world],
        'route_sample_statuses':details,
        'lane_indices': [c['lane_index'] for c in chosen],
        'lanes_per_segment':[len(c['lanes']) for c in chosen],
        'start_lane_index':chosen[0]['lane_index'],
        'selected_lane_gaps':gaps,
        'complete':all(i['complete'] for i in details),
        'lane_chain_continuous':len(gaps)==0,
        'usable_for_shadow':good,
        'note':'Geometry nearest-lane continuity, not a semantic match to PDD or a required turning lane.'
    }
