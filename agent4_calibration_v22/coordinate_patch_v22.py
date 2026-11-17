"""Versioned *shadow-only* coordinate correction for MetaDrive V2 feature adapter.

Never apply model predictions directly: waypoint horizon and action gains differ
between PDD CARLA and MetaDrive.

For installed MetaDrive we detect whether ego-local positive Y is LEFT or RIGHT
by projecting a world-left unit vector using the vehicle's own conversion API.
This measured sign is used for navigation lateral coordinates and to relabel
left/right surrounding vehicle sectors from the V2 candidate adapter.
"""
from __future__ import annotations
import math
from typing import Mapping

LATERAL_PAIRS = (
    ('left_vehicle_dist_m', 'right_vehicle_dist_m'),
    ('has_left_vehicle', 'has_right_vehicle'),
)


def determine_local_y_to_right_multiplier(agent) -> tuple[int, dict]:
    """Return +1 if local Y points RIGHT, -1 if local Y points LEFT.

    Independently tests conversion at current heading: world-left vector
    [-sin(heading), cos(heading)] must project to lateral +/-1.
    """
    theta = float(agent.heading_theta)
    if not math.isfinite(theta):
        raise ValueError('Invalid heading_theta')
    left_vector = [-math.sin(theta), math.cos(theta)]
    loc = agent.convert_to_local_coordinates(left_vector, 0.0)
    local_forward, local_lateral = map(float, loc[:2])
    if not all(math.isfinite(v) for v in (local_forward, local_lateral)):
        raise ValueError('Coordinate conversion produced non-finite values')
    # Tight tolerance ensures actual basis convention is identified and not assumed.
    if abs(local_forward) > 0.12 or not 0.8 < abs(local_lateral) < 1.2:
        raise RuntimeError(
            'Cannot establish MetaDrive ego-local lateral axis: '
            f'world-left -> local [{local_forward}, {local_lateral}]'
        )
    # World left projects to +1 => local positive Y=left, so negate to get right.
    multiplier = -1 if local_lateral > 0 else 1
    return multiplier, {
        'heading_rad': theta,
        'world_left_unit_xy': left_vector,
        'world_left_in_ego_local_xy': [local_forward, local_lateral],
        'local_y_positive': 'left' if multiplier == -1 else 'right',
        'raw_to_pdd_right_multiplier': multiplier,
    }


def correct_features_to_pdd_right(raw: Mapping[str, float],
                                  y_to_right_multiplier: int) -> dict[str, float]:
    """Convert V2 MetaDrive features into common +right lateral convention.

    V2 code assumes local Y is right when classifying nearby cars. If local Y
    is actually left, swap both side distances and presence masks.
    """
    if y_to_right_multiplier not in (-1, 1):
        raise ValueError('Multiplier must be ±1')
    out = dict(raw)
    for k in ('wp1_right_m', 'wp2_right_m'):
        out[k] = float(out[k]) * y_to_right_multiplier
    if y_to_right_multiplier == -1:
        for left, right in LATERAL_PAIRS:
            out[left], out[right] = float(raw[right]), float(raw[left])
    return out


def carla_to_metadrive_steer(carla_steer: float, gain: float = 1.0) -> float:
    """PROVISIONAL sign-only transform, assuming CARLA +right, MetaDrive +left.

    gain=1 is NOT physics calibration. Used for shadow logging, NEVER auto-drive.
    """
    raw = -float(carla_steer) * float(gain)
    if not math.isfinite(raw):
        raise ValueError('Invalid steering prediction')
    return max(-1.0, min(1.0, raw))
