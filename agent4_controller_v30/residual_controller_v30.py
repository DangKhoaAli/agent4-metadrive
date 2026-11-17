"""Stateless, bounded residual steering for MetaDrive simulation.

All angles are radians. Heading error is vehicle heading minus lane tangent
(counterclockwise positive). Lateral error is positive toward the physical right
of the planned lane. MetaDrive positive steering turns physically left.

This controller does NOT learn or retrain anything: it adds a conventional
lateral/heading feedback residual to the PDD model's steering prediction.
"""
from __future__ import annotations

import math
from typing import Optional


def finite(name, value):
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be finite, got {value!r}')
    if not math.isfinite(x):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return x


def clip(value, low, high):
    return max(float(low), min(float(high), finite('value', value)))


def wrap_angle(value):
    value = finite('angle', value)
    return math.atan2(math.sin(value), math.cos(value))


def lateral_sign_from_lane_geometry(lane, longitude):
    """+1 if lane.local_coordinates lateral is +right, -1 if +left.

    Uses lane.position(s, lat=+0.25) and the tangent as a sign certificate,
    not a CARLA/MetaDrive coordinate convention guess.
    """
    length = finite('lane length', lane.length)
    if length <= 0.5:
        raise ValueError('Lane too short to calibrate lateral axis')
    s = clip(longitude, 0.25, length - 0.25)
    p_minus = lane.position(s - 0.2, 0.0)
    p_plus = lane.position(s + 0.2, 0.0)
    p_center = lane.position(s, 0.0)
    p_offset = lane.position(s, 0.25)
    tx = float(p_plus[0]) - float(p_minus[0])
    ty = float(p_plus[1]) - float(p_minus[1])
    norm = math.hypot(tx, ty)
    if norm < 1e-5:
        raise ValueError('Degenerate lane tangent')
    # Physical right is +sin(heading), -cos(heading).
    right_x, right_y = ty / norm, -tx / norm
    dx = float(p_offset[0]) - float(p_center[0])
    dy = float(p_offset[1]) - float(p_center[1])
    projection = dx * right_x + dy * right_y
    if abs(projection) < 0.05:
        raise ValueError('Ambiguous lane lateral sign')
    return 1 if projection > 0 else -1


def rightward_lateral_error(lane, position):
    """Return right-positive lateral distance, segment longitude and sign."""
    s, lateral = lane.local_coordinates(position)
    sign = lateral_sign_from_lane_geometry(lane, s)
    return finite('lateral', lateral) * sign, float(s), sign


def compute_correction(lateral_right_m, heading_error_rad, kp_lateral=0.035,
                       kp_heading=0.30, residual_limit=0.08):
    """Positive steer = LEFT: correct right-of-lane with +steer,
    correct left-facing heading error with -steer.

    NOTE: heuristic gains for A/B experiments, NOT physically calibrated.
    """
    lateral = finite('lateral_right_m', lateral_right_m)
    heading = wrap_angle(heading_error_rad)
    kl = finite('kp_lateral', kp_lateral)
    kh = finite('kp_heading', kp_heading)
    limit = finite('residual_limit', residual_limit)
    if kl < 0 or kh < 0 or not 0 < limit <= 0.35:
        raise ValueError('Invalid residual gains/limit')
    raw = kl * lateral - kh * heading
    bounded = clip(raw, -limit, limit)
    return {'raw': raw, 'bounded': bounded,
            'lateral_term': kl * lateral, 'heading_term': -kh * heading}


def steering_action(model_carla_steer, mode='baseline', lateral_right_m=0.0,
                    heading_error_rad=0.0, kp_lateral=0.035, kp_heading=0.30,
                    residual_limit=0.08, gain=1.0, steer_limit=0.35,
                    previous: Optional[float]=None, max_delta=0.12):
    if mode not in {'baseline', 'residual'}:
        raise ValueError('mode must be baseline or residual')
    base = -finite('model_carla_steer', model_carla_steer) * finite('gain', gain)
    correction = compute_correction(lateral_right_m, heading_error_rad,
                                    kp_lateral, kp_heading, residual_limit) if mode == 'residual' else {
            'raw': 0.0, 'bounded': 0.0, 'lateral_term': 0.0, 'heading_term': 0.0}
    combined = base + correction['bounded']
    saturated = clip(combined, -steer_limit, steer_limit)
    if previous is not None:
        prev = finite('previous steer', previous)
        saturated = clip(saturated, prev-max_delta, prev+max_delta)
    return {'steer': saturated, 'base': base, 'correction': correction,
            'combined_before_limits': combined,
            'steering_saturation': abs(combined) > steer_limit + 1e-10,
            'slew_limited': previous is not None and abs(saturated - clip(combined, -steer_limit, steer_limit)) > 1e-10}


def estimated_edge_clearance_m(vehicle, heading_error_rad):
    """Conservative, *approximate* lateral clearance of rotated vehicle footprint.

    MetaDrive flag/collision geometry is authoritative. Distances are to road
    edges reported by MetaDrive; this approximation is not a certified guard.
    """
    w = finite('WIDTH', vehicle.WIDTH)
    l = finite('LENGTH', vehicle.LENGTH)
    left = finite('dist_to_left_side', vehicle.dist_to_left_side)
    right = finite('dist_to_right_side', vehicle.dist_to_right_side)
    angle = abs(wrap_angle(heading_error_rad))
    envelope = 0.5*w*abs(math.cos(angle)) + 0.5*l*abs(math.sin(angle))
    return {'left': left-envelope, 'right': right-envelope,
            'minimum': min(left, right)-envelope, 'envelope': envelope}
