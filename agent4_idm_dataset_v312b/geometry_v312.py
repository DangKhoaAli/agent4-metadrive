"""Read-only route-geometry diagnostics for MetaDrive V3.12.

All measurements are proxies, never vehicle safety certificates.
No steering or throttle is produced here. Works without importing MetaDrive.
"""
from __future__ import annotations

import math


def finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def lane_id(lane):
    if lane is None:
        return None
    return repr(getattr(lane, 'index', None))


def safe_nav_lanes(nav, attr):
    value = getattr(nav, attr, None) if nav is not None else None
    return [x for x in (value or []) if x is not None]


def lane_heading(lane, s):
    length = finite(getattr(lane, 'length', None))
    if length is None or length < .05:
        raise ValueError('Invalid lane length')
    s = max(0., min(float(s), length))
    start = max(0., s - .20)
    end = min(length, s + .20)
    if end - start < 1e-5:
        raise ValueError('Lane heading tangent degenerate')
    p0, p1 = lane.position(start, 0.), lane.position(end, 0.)
    dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
    if math.hypot(dx, dy) < 1e-5:
        raise ValueError('Lane heading positions degenerate')
    return math.atan2(dy, dx)


def lane_pose(lane, position, vehicle_heading):
    """Right-positive deviation, using lane geometry to certify lateral sign."""
    s, raw_lat = lane.local_coordinates(position)
    s, raw_lat = float(s), float(raw_lat)
    length = float(lane.length)
    if length <= .5:
        raise ValueError('Lane too short')
    at = max(.25, min(s, length - .25))
    p0 = lane.position(at, 0.)
    p1 = lane.position(at, .25)
    tangent = lane_heading(lane, at)
    # Given lane direction, physical right unit vector = (sin theta, -cos theta).
    d_right = ((float(p1[0]) - float(p0[0])) * math.sin(tangent)
               - (float(p1[1]) - float(p0[1])) * math.cos(tangent))
    if abs(d_right) < .05:
        raise ValueError('Lane lateral-axis sign ambiguous')
    sign = 1. if d_right > 0 else -1.
    current_heading = lane_heading(lane, s)
    remaining = max(0., length - s)
    ahead = min(length, max(0., s) + 8.0)
    ahead_heading = lane_heading(lane, ahead)
    distance = max(0.01, ahead - max(0., min(length, s)))
    return dict(
        lane_class=type(lane).__name__, lane_index=lane_id(lane),
        lane_longitudinal_m=s, lane_length_m=length,
        distance_to_lane_end_m=remaining,
        lateral_right_m=raw_lat * sign,
        heading_error_rad=wrap_angle(float(vehicle_heading) - current_heading),
        lane_heading_rad=current_heading,
        lane_heading_change_ahead_rad=wrap_angle(ahead_heading - current_heading),
        lane_curvature_proxy_rad_per_m=wrap_angle(ahead_heading-current_heading)/distance,
        lateral_sign_proven_by_geometry=sign,
    )


def compare_vehicle_route(agent):
    """Separate physical lane and planned lane. Route metadata may not exist."""
    position = agent.position
    theta = float(agent.heading_theta)
    nav = getattr(agent, 'navigation', None)
    current = safe_nav_lanes(nav, 'current_ref_lanes')
    upcoming = safe_nav_lanes(nav, 'next_ref_lanes')
    actual = getattr(agent, 'lane', None)
    # Pick current planned lane closest to the ego, not arbitrary lane index 0.
    route = None
    if current:
        def route_distance(candidate):
            try:
                return abs(float(candidate.local_coordinates(position)[1]))
            except (AttributeError, TypeError, ValueError):
                return float('inf')
        route = min(current, key=route_distance)
    out = {
        'vehicle_lane_id': lane_id(actual),
        'vehicle_lane_class': type(actual).__name__ if actual is not None else None,
        'planned_lane_id': lane_id(route),
        'planned_lane_class': type(route).__name__ if route is not None else None,
        'planned_next_lane_id': lane_id(upcoming[0]) if upcoming else None,
        'planned_next_lane_class': type(upcoming[0]).__name__ if upcoming else None,
        'vehicle_lane_differs_from_plan': (
            lane_id(actual) != lane_id(route) if actual is not None and route is not None else None),
        'route_geometry_available': bool(route),
        'road_geometry_errors': [],
    }
    for prefix, lane in (('physical', actual), ('planned', route)):
        if lane is None:
            continue
        try:
            details = lane_pose(lane, position, theta)
            out[prefix] = details
        except (ValueError, TypeError, AttributeError, IndexError, KeyError) as ex:
            out['road_geometry_errors'].append(f'{prefix}: {type(ex).__name__}: {ex}')
    # This is a heading jump between successive geometries, NOT an inferred curvature
    # if the maps' actual centerlines are discontinuous.
    if route is not None and upcoming:
        try:
            end = lane_heading(route, float(route.length))
            start = lane_heading(upcoming[0], 0.)
            out['route_heading_jump_at_transition_rad'] = wrap_angle(start - end)
        except (ValueError, TypeError, AttributeError, IndexError) as ex:
            out['road_geometry_errors'].append(f'route transition: {type(ex).__name__}: {ex}')
    return out


def approximate_vehicle_clearance(agent, planned_heading_error):
    """Proxy only; uses the vehicle rectangle and MetaDrive side distances."""
    try:
        width = float(agent.WIDTH)
        length = float(agent.LENGTH)
        left = float(agent.dist_to_left_side)
        right = float(agent.dist_to_right_side)
        angle = abs(float(planned_heading_error))
        radius = .5 * width * abs(math.cos(angle)) + .5 * length * abs(math.sin(angle))
        result = min(left, right) - radius
        return result if math.isfinite(result) else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
