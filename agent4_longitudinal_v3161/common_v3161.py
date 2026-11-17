"""V3.16.1: stateful, learned-model-only gas/brake output stabilization.

The frozen V3.16 estimator supplies the ONLY action signal to the candidate.
This signal-processing experiment does not retrain a model and has no rule or IDM fallback.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path

VERSION = 'AGENT4_V3161_STABILIZED_LONGITUDINAL'

@dataclass(frozen=True)
class FilterConfig:
    # Greater down_rate allows commanded braking after speed-target changes.
    up_rate: float = 0.30
    down_rate: float = 0.65
    max_step_up: float = 0.16
    max_step_down: float = 0.35
    deadband: float = 0.065
    max_throttle: float = 0.70
    max_brake: float = 0.80

    def __post_init__(self):
        vals=asdict(self)
        if not all(isinstance(v,(int,float)) and math.isfinite(float(v)) for v in vals.values()):
            raise ValueError('Nonfinite stabilization setting')
        if not 0 < self.up_rate <= 1 or not 0 < self.down_rate <= 1:
            raise ValueError('Filter rates must be (0,1]')
        if not 0 < self.max_step_up <= 1 or not 0 < self.max_step_down <= 1:
            raise ValueError('Max action changes must be (0,1]')
        if not 0 <= self.deadband < .20:
            raise ValueError('Deadband must be [0,0.2)')
        if not .1 <= self.max_throttle <= 1 or not .1 <= self.max_brake <= 1:
            raise ValueError('Action caps must be within [0.1,1]')
        if self.deadband >= min(self.max_throttle,self.max_brake):
            raise ValueError('Deadband must be smaller than both caps')

    def to_dict(self):
        return asdict(self)


class StabilizedLongitudinal:
    """Apply a causal low-pass/slew-rate filter to the *learned* signed action.

    - No access to the rule governor, teacher, future state or hidden labels.
    - State reset exactly once per new MetaDrive episode.
    - First command can accelerate from standstill, but remains capped.
    """
    def __init__(self, config: FilterConfig | None = None):
        self.config=config or FilterConfig()
        self.reset()

    def reset(self):
        self._previous=None

    def apply(self, predicted: float) -> float:
        raw=float(predicted)
        if not math.isfinite(raw):
            raise ValueError('Nonfinite learned gas/brake prediction')
        if raw < -1.0001 or raw > 1.0001:
            raise ValueError('Learned gas/brake prediction outside [-1,1]')
        c=self.config
        clipped=max(-c.max_brake,min(c.max_throttle,raw))
        # Suppress tiny signed commands; not a target-speed controller.
        if abs(clipped)<c.deadband:
            clipped=0.0
        if self._previous is None:
            result=clipped
        else:
            prev=self._previous
            factor=c.down_rate if clipped<prev else c.up_rate
            desired=prev+factor*(clipped-prev)
            change=max(-c.max_step_down,min(c.max_step_up,desired-prev))
            result=prev+change
            result=max(-c.max_brake,min(c.max_throttle,result))
            if abs(result)<c.deadband:
                result=0.0
        self._previous=result
        return result


def step_from_model(filter_controller, bundle, row, feat):
    """Import the original V3.16 predictor via its original common module."""
    from common_v316 import predict_long
    return filter_controller.apply(predict_long(bundle,row,feat))


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
