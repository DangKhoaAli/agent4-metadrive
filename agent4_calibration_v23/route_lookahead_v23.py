"""Experimental route-aware waypoint sampling for MetaDrive 0.4.3.

NOT a training-time adapter and NOT certified equivalent to CARLA PDD.
Use the route's road/lane geometry to sample points ~25/45 m ahead
along the road instead of selecting navigation's nearest node checkpoint.

Uses privileged simulator map/route state. No learned action is applied.
"""
from __future__ import annotations

import math
from typing import Any, Sequence


def _finite_float(x, name):
    v = float(x)
    if not math.isfinite(v):
        raise ValueError('%s must be finite, got %s' % (name, x))
    return v


def road_center_point(lanes: Sequence[Any], longitude: float):
    """Replicate MetaDrive 0.4.3 NodeNetworkNavigation checkpoint center.

    Per source: first reference lane + lateral_offset=(num_lanes/2-0.5)*width.
    Lane.position(longitudinal, lateral) handles straight AND curved segments.
    """
    if not lanes:
        raise ValueError('Empty reference lane list')
    lane = lanes[0]
    width = _finite_float(lane.width, 'lane width')
    lateral_middle = (len(lanes) / 2.0 - 0.5) * width
    world = lane.position(float(longitude), lateral_middle)
    x, y = _finite_float(world[0], 'world.x'), _finite_float(world[1], 'world.y')
    return (x, y)


def collect_forward_segments(nav, agent):
    """Use planned road checkpoints, not nearest road from geometry alone.

    Returns [(lanes, start_longitude), ...] for route from current segment onward.
    """
    checkpoints = list(getattr(nav, 'checkpoints'))
    indices = list(getattr(nav, '_target_checkpoints_index'))
    if len(checkpoints) < 2 or not indices:
        raise ValueError('No planned route checkpoints')
    start_idx = int(indices[0])
    if not 0 <= start_idx < len(checkpoints)-1:
        raise ValueError('Invalid current route segment index')
    graph = nav.map.road_network.graph
    segments = []
    for j in range(start_idx, len(checkpoints)-1):
        a, b = checkpoints[j:j+2]
        try:
            lanes = graph[a][b]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError('Missing planned road %r -> %r' % (a, b)) from exc
        if not lanes:
            raise ValueError('No reference lane on road %r -> %r' % (a, b))
        lane = lanes[0]
        lane_length = _finite_float(lane.length, 'lane.length')
        if lane_length <= 0:
            raise ValueError('Invalid lane length')
        if j == start_idx:
            projected_long, _ = lane.local_coordinates(agent.position)
            projected_long = _finite_float(projected_long, 'current projection')
            projected_long = max(0.0, min(projected_long, lane_length))
        else:
            projected_long = 0.0
        segments.append({'lanes':lanes, 'start_longitude': projected_long,
                         'length':lane_length, 'nodes':(a,b)})
    return segments


def _sample_at_distance(segments, distance_m):
    target = _finite_float(distance_m, 'lookahead')
    if target < 0.0:
        raise ValueError('lookahead must be >= 0')
    remaining = target
    travelled = 0.0
    for segment in segments:
        length = segment['length']
        start = segment['start_longitude']
        available = max(0.0, length - start)
        if remaining <= available + 1e-7:
            longitude = min(length, start + max(remaining,0.0))
            return road_center_point(segment['lanes'], longitude), {
                'requested_m':target, 'achieved_m':travelled + (longitude-start),
                'complete':True, 'road': list(map(str, segment['nodes'])),
            }
        remaining -= available
        travelled += available
    last = segments[-1]
    world = road_center_point(last['lanes'], last['length'])
    return world, {
        'requested_m':target, 'achieved_m':travelled,
        'complete':False, 'road':list(map(str, last['nodes'])),
    }


def sample_route_waypoints(env, horizons=(25.0,45.0), y_to_right_multiplier=-1,
                           max_centerline_jump_m=2.0):
    """Return route-aware ego-relative waypoints with +y=RIGHT.

    Aborts neither at lane discontinuity nor route-end; reports both. Callers
    should NOT use outputs for a driving policy when quality flags are false.
    """
    if len(horizons) != 2 or horizons[0] <= 0 or horizons[1] <= horizons[0]:
        raise ValueError('Expected two increasing, positive lookahead distances')
    if y_to_right_multiplier not in (-1,1):
        raise ValueError('y_to_right_multiplier must be +/-1')
    agent = env.agent
    nav = agent.navigation
    segments = collect_forward_segments(nav, agent)
    jumps = []
    for former, latter in zip(segments[:-1], segments[1:]):
        end = road_center_point(former['lanes'], former['length'])
        start = road_center_point(latter['lanes'], 0.0)
        gap = math.hypot(end[0]-start[0], end[1]-start[1])
        if gap > max_centerline_jump_m:
            jumps.append({'between': [list(map(str, former['nodes'])),list(map(str,latter['nodes']))],
                          'gap_m':round(gap,6)})
    wp = []
    statuses = []
    world_points = []
    pos = [_finite_float(agent.position[0],'ego_x'), _finite_float(agent.position[1],'ego_y')]
    for distance in horizons:
        world, diag = _sample_at_distance(segments, distance)
        local_x, local_y = agent.convert_to_local_coordinates(
            (world[0]-pos[0], world[1]-pos[1]), 0.0)
        x = _finite_float(local_x, 'local_x')
        y = _finite_float(local_y, 'local_y') * y_to_right_multiplier
        wp.append((x,y)); world_points.append(world); statuses.append(diag)
    return wp, {
        'requested_lookahead_m': list(map(float, horizons)),
        'route_waypoints_world_xy': [list(p) for p in world_points],
        'route_sample_statuses': statuses,
        'num_remaining_segments': len(segments),
        'centerline_discontinuities': jumps,
        'complete':all(i['complete'] for i in statuses),
        'geometry_continuous':len(jumps)==0,
        'usable_for_shadow': all(i['complete'] for i in statuses) and not jumps,
        'note':'Fixed road-arc-distance lookahead, not equal to original CARLA route target selection.',
    }
