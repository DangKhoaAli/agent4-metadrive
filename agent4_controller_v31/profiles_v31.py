"""Fixed, auditable V3.1 MetaDrive-only A/B experiment profiles.

These are experimental rule-based residual gains, not learned parameters.
No setting changes the authoritative MetaDrive out_of_road predicate.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from types import SimpleNamespace

@dataclass(frozen=True)
class Profile:
    mode: str
    speed_limit_mps: float
    kp_lateral: float
    kp_heading: float
    residual_limit: float
    note: str


PROFILES = {
    'baseline_3': Profile('baseline', 3.0, 0.035, 0.30, 0.08,
                          'Exact V3.0 model-only steering reference at 3 m/s'),
    'v30_3': Profile('residual', 3.0, 0.035, 0.30, 0.08,
                     'Exact V3.0 residual steering reference at 3 m/s'),
    'heading_3': Profile('residual', 3.0, 0.035, 0.60, 0.08,
                         'Increase heading correction, keep lateral correction fixed'),
    'lateral_3': Profile('residual', 3.0, 0.080, 0.30, 0.08,
                         'Increase lateral correction, keep heading correction fixed'),
    'balanced_3': Profile('residual', 3.0, 0.070, 0.60, 0.12,
                          'Increase both corrections; exploratory only'),
    'baseline_2': Profile('baseline', 2.0, 0.035, 0.30, 0.08,
                          'Unmodified model steering at reduced speed'),
    'v30_2': Profile('residual', 2.0, 0.035, 0.30, 0.08,
                     'V3.0 residual steering at reduced speed'),
}
DEFAULT_PROFILES = 'baseline_3,v30_3,heading_3,lateral_3,baseline_2,v30_2'


def select_profiles(csv):
    selected = [x.strip() for x in csv.split(',') if x.strip()]
    if not selected:
        raise ValueError('At least one profile must be selected')
    if len(set(selected)) != len(selected):
        raise ValueError('Profiles must not repeat')
    unknown = sorted(set(selected) - set(PROFILES))
    if unknown:
        raise ValueError(f'Unknown profiles: {unknown}; available: {sorted(PROFILES)}')
    return selected


def options_for(profile: Profile, *, steps=1400, trace_stride=10, tail_steps=35):
    """Match all V3.0 non-experimental defaults, vary only profile parameters."""
    if steps < 10 or not 1 <= trace_stride <= 500 or not 1 <= tail_steps <= 200:
        raise ValueError('Invalid steps or trace settings')
    return SimpleNamespace(
        steps=steps,
        speed_limit_mps=profile.speed_limit_mps,
        steer_gain=1.0,
        kp_lateral=profile.kp_lateral,
        kp_heading=profile.kp_heading,
        residual_limit=profile.residual_limit,
        steer_limit=0.35,
        max_steer_delta=0.12,
        guard_lane_offset_m=1.5,
        guard_heading_rad=0.45,
        guard_road_margin_m=0.0,  # Keep MetaDrive line-contact termination authoritative.
        warn_road_margin_m=0.35,
        max_gap=2.0,
        trace_stride=trace_stride,
        tail_steps=tail_steps,
    )


def profile_dict(profile):
    return asdict(profile)


def episode_summary(result):
    info = result.get('end_info') or {}
    trace = result.get('trace') or []
    terminal = result.get('terminal_road_state') or {}
    return {
        'status':result.get('status'),
        'route_completion':info.get('route_completion'),
        'travelled_approx_m':result.get('travelled_approx_m'),
        'n_applied_steps':result.get('n_applied_steps'),
        'max_abs_planned_lateral_right_m':result.get('max_abs_planned_lateral_right_m'),
        'max_abs_heading_error_rad':result.get('max_abs_heading_error_rad'),
        'min_estimated_edge_clearance_m':result.get('min_estimated_edge_clearance_m'),
        'near_edge_warning_steps':result.get('near_edge_warning_steps'),
        'correction_active_steps':result.get('correction_active_steps'),
        'steering_saturation_steps':result.get('steering_saturation_steps'),
        'steering_slew_limited_steps':result.get('steering_slew_limited_steps'),
        'terminal_reasons':terminal.get('active_reasons',[]),
        'error_count':len(result.get('errors') or []),
        'terminal_sample':({k:trace[-1].get(k) for k in
                            ('step','planned_lateral_right_m','heading_error_rad',
                             'model_steer_md','residual_steer','action_applied',
                             'road_margin_approx_m','road_causes_after')}
                           if trace else None),
    }
