"""Shared **candidate** PDD/MetaDrive observation feature contract (Python 3.10).

This file provides two adapters with the same physical units and feature order.
MetaDrive 0.4.3 compatibility must be verified by probe_metadrive_v2.py on user PC.
It intentionally does not validate simulator dynamics, camera domain, or timestamps.

Coordinate convention: +x forward, +y vehicle-right. Distances are object
*centre* distances (not bumper gaps). Absence = DIST_CAP together with a mask.
"""
from __future__ import annotations

import math
from typing import Any

DIST_CAP = 60.0
WAYPOINT_CAP = 50.0

FEATURES = [
    'speed_mps',
    'wp1_forward_m', 'wp1_right_m',
    'wp2_forward_m', 'wp2_right_m',
    'lane_width_m',
    'front_vehicle_dist_m', 'front_vehicle_closing_speed_mps',
    'left_vehicle_dist_m', 'right_vehicle_dist_m',
    'front_pedestrian_dist_m',
    'has_front_vehicle', 'has_left_vehicle',
    'has_right_vehicle', 'has_front_pedestrian',
]


class AdapterDataError(ValueError):
    pass


def number(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as ex:
        raise AdapterDataError(f'{name}: non-numeric value {value!r}') from ex
    if not math.isfinite(value):
        raise AdapterDataError(f'{name}: non-finite value {value!r}')
    return value


def cap_waypoint(x: Any, y: Any) -> tuple[float, float]:
    x, y = number(x, 'waypoint.x'), number(y, 'waypoint.y')
    r = math.hypot(x, y)
    if r > WAYPOINT_CAP:
        return x * WAYPOINT_CAP / r, y * WAYPOINT_CAP / r
    return x, y


def _object_type(label: str) -> str | None:
    s = str(label).lower().strip()
    if s in ('ego_car', 'ego', 'self'):
        return None
    if s in ('car', 'vehicle', 'truck', 'bus', 'motorcycle', 'bike', 'bicycle') or s.startswith('vehicle.'):
        return 'vehicle'
    if s in ('walker', 'pedestrian', 'person') or s.startswith('walker.') or s.startswith('pedestrian'):
        return 'pedestrian'
    return None


def _empty_objects() -> dict[str, float]:
    return {
        'front_vehicle_dist_m': DIST_CAP,
        'front_vehicle_closing_speed_mps': 0.0,
        'left_vehicle_dist_m': DIST_CAP,
        'right_vehicle_dist_m': DIST_CAP,
        'front_pedestrian_dist_m': DIST_CAP,
        'has_front_vehicle': 0.0,
        'has_left_vehicle': 0.0,
        'has_right_vehicle': 0.0,
        'has_front_pedestrian': 0.0,
    }


def summarize_objects(objects: list[dict[str, Any]], ego_speed_mps: float,
                      y_sign: int = 1, corridor_half_width: float = 2.5) -> dict[str, float]:
    """Find nearest objects in fixed ego-centric sectors.

    Objects: {'kind': 'vehicle'|'pedestrian', 'x': forward metres,
              'y': right metres, 'speed_mps': nonnegative scalar or None}.
    Same algorithm is applied to both simulators to minimize feature mismatch.
    A longitudinal margin of -10 m for side sectors includes vehicles abeam.
    """
    if y_sign not in (-1, 1):
        raise AdapterDataError('y_sign must be +1 or -1')
    out = _empty_objects()
    for ob in objects:
        kind = ob.get('kind')
        if kind not in ('vehicle', 'pedestrian'):
            continue
        x = number(ob['x'], 'object.x')
        y = y_sign * number(ob['y'], 'object.y')
        dist = math.hypot(x, y)
        if dist > DIST_CAP or dist < 0.001:
            continue
        if kind == 'vehicle':
            if 0 <= x <= DIST_CAP and abs(y) <= corridor_half_width:
                if dist < out['front_vehicle_dist_m']:
                    out['front_vehicle_dist_m'] = dist
                    obj_speed = ob.get('speed_mps')
                    if obj_speed is not None:
                        speed = number(obj_speed, 'object.speed_mps')
                        # Relative velocity magnitude proxy; NOT relative longitudinal velocity.
                        out['front_vehicle_closing_speed_mps'] = max(-40.0, min(40.0, ego_speed_mps - speed))
                    else:
                        out['front_vehicle_closing_speed_mps'] = 0.0
                    out['has_front_vehicle'] = 1.0
            elif -10 <= x <= 40 and corridor_half_width < y <= 12:
                if dist < out['right_vehicle_dist_m']:
                    out['right_vehicle_dist_m'] = dist
                    out['has_right_vehicle'] = 1.0
            elif -10 <= x <= 40 and -12 <= y < -corridor_half_width:
                if dist < out['left_vehicle_dist_m']:
                    out['left_vehicle_dist_m'] = dist
                    out['has_left_vehicle'] = 1.0
        else:
            if 0 <= x <= DIST_CAP and abs(y) <= corridor_half_width:
                if dist < out['front_pedestrian_dist_m']:
                    out['front_pedestrian_dist_m'] = dist
                    out['has_front_pedestrian'] = 1.0
    return out


def pack_features(speed_mps: float, wp1: tuple[float, float], wp2: tuple[float, float],
                  lane_width_m: float, objects: list[dict[str, Any]], y_sign: int = 1) -> dict[str, float]:
    speed = number(speed_mps, 'speed_mps')
    lane = number(lane_width_m, 'lane_width_m')
    if speed < 0 or speed > 90:
        raise AdapterDataError(f'speed_mps out of expected range: {speed}')
    if not (1.5 <= lane <= 8.0):
        raise AdapterDataError(f'lane_width_m out of expected range: {lane}')
    wp1 = cap_waypoint(*wp1)
    wp2 = cap_waypoint(*wp2)
    out = {
        'speed_mps': speed,
        'wp1_forward_m': wp1[0], 'wp1_right_m': wp1[1],
        'wp2_forward_m': wp2[0], 'wp2_right_m': wp2[1],
        'lane_width_m': lane,
    }
    out.update(summarize_objects(objects, speed, y_sign=y_sign))
    assert set(out) == set(FEATURES), 'Feature schema bug'
    return out


def pdd_features(info: dict[str, Any], raw_boxes: Any, box_y_sign: int = 1,
                 waypoint_y_sign: int = 1) -> dict[str, float]:
    """PDD boxes per dataset card are ego-relative [forward, right?, up].

    CARLA-side coordinate sign is configurable and MUST be checked on local data.
    """
    lane = info.get('lane_info', {})
    if not isinstance(lane, dict):
        raise AdapterDataError('PDD lane_info missing or not a dict')
    width = lane.get('lane_width')
    if not isinstance(raw_boxes, list):
        raise AdapterDataError(f'PDD boxes expected a list, got {type(raw_boxes).__name__}')
    objects = []
    for ob in raw_boxes:
        if not isinstance(ob, dict):
            continue
        kind = _object_type(ob.get('class', ''))
        if kind is None:
            continue
        pos = ob.get('position')
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            continue
        objects.append({
            'kind': kind, 'x': pos[0], 'y': pos[1],
            'speed_mps': ob.get('speed', None),
        })
    wp1 = info.get('target_point')
    wp2 = info.get('target_point_next')
    if not isinstance(wp1, (tuple, list)) or len(wp1) < 2 or not isinstance(wp2, (tuple, list)) or len(wp2) < 2:
        raise AdapterDataError('PDD target_point fields are absent or malformed')
    return pack_features(
        info.get('speed'), (wp1[0], number(wp1[1], 'PDD wp1.y') * waypoint_y_sign),
        (wp2[0], number(wp2[1], 'PDD wp2.y') * waypoint_y_sign), width,
        objects, y_sign=box_y_sign,
    )


def metadrive_features(env, vehicle_y_sign: int = 1) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract physical ego-centric features from a live MetaDriveEnv 0.4.3.

    Uses navigation.get_navi_info() (first/second checkpoint XY), NOT the full
    default state vector and NOT the raw lidar rays. This is a *privileged*
    state adapter, which is fine in simulation but not a camera-only policy.
    """
    agent = env.agent
    nav = agent.navigation
    if nav is None:
        raise AdapterDataError('MetaDrive navigation unavailable')
    nav_vec = list(nav.get_navi_info())
    if len(nav_vec) < 7:
        raise AdapterDataError(f'MetaDrive nav length {len(nav_vec)} <7')
    dist_scale = number(getattr(nav, 'NAVI_POINT_DIST'), 'NAVI_POINT_DIST')
    wp1 = ((number(nav_vec[0], 'nav[0]') * 2 - 1) * dist_scale,
           (number(nav_vec[1], 'nav[1]') * 2 - 1) * dist_scale * vehicle_y_sign)
    wp2 = ((number(nav_vec[5], 'nav[5]') * 2 - 1) * dist_scale,
           (number(nav_vec[6], 'nav[6]') * 2 - 1) * dist_scale * vehicle_y_sign)
    lane = getattr(agent, 'lane', None)
    if lane is None:
        raise AdapterDataError('MetaDrive agent.lane unavailable')
    lane_width = getattr(lane, 'width', None)
    speed = number(agent.speed_km_h, 'speed_km_h') / 3.6

    # Trust vehicle class checks to avoid counting map / static engine objects.
    try:
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
    except ImportError as ex:
        raise AdapterDataError('Cannot import MetaDrive BaseVehicle') from ex
    all_objects = env.engine.get_objects()
    if not isinstance(all_objects, dict):
        raise AdapterDataError('engine.get_objects() did not return a dict')
    objects = []
    num_other_vehicles = 0
    num_pedestrians = 0
    for ob in all_objects.values():
        if ob is agent:
            continue
        if isinstance(ob, BaseVehicle):
            kind = 'vehicle'
            num_other_vehicles += 1
        elif 'pedestrian' in type(ob).__name__.lower() or 'walker' in type(ob).__name__.lower():
            kind = 'pedestrian'
            num_pedestrians += 1
        else:
            continue
        pos = getattr(ob, 'position', None)
        if pos is None or len(pos) < 2:
            continue
        delta = [number(pos[0], 'other.pos[0]') - number(agent.position[0], 'ego.pos[0]'),
                 number(pos[1], 'other.pos[1]') - number(agent.position[1], 'ego.pos[1]')]
        x_local, y_local = agent.convert_to_local_coordinates(delta, 0.0)
        speed_kmh = getattr(ob, 'speed_km_h', None)
        objects.append({
            'kind': kind, 'x': x_local, 'y': y_local,
            'speed_mps': number(speed_kmh, 'other.speed_kmh') / 3.6 if speed_kmh is not None else None,
        })
    features = pack_features(speed, wp1, wp2, lane_width, objects)
    diagnostics = {
        'n_engine_objects': len(all_objects),
        'n_other_vehicles': num_other_vehicles,
        'n_pedestrians': num_pedestrians,
        'n_candidate_objects': len(objects),
        'nav_distance_cap_m': dist_scale,
        'raw_nav_xy': [float(nav_vec[i]) for i in (0, 1, 5, 6)],
        'ego_position': [float(v) for v in agent.position[:2]],
        'ego_heading_rad': float(agent.heading_theta),
        'note': 'MetaDrive traffic info here comes from simulator object states (privileged observation).',
    }
    return features, diagnostics
