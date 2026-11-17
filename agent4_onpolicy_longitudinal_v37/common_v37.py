"""V3.7: governor-teacher longitudinal behavior cloning, V3.5 steering on-policy."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'agent4_longitudinal_v36'))
from common_v36 import names, make_row, predict_action, check_splits, finite

VERSION = 'V37_V35_STEERING_GOVERNOR_TEACHER_LONGITUDINAL'
DATA_STATUS = 'V37_ON_POLICY_GOVERNOR_TEACHER_DATA'


def validate_new_splits(groups):
    check_splits(groups['train'], groups['val'], groups['eval'])
    for split in ('train', 'val', 'eval'):
        if min(groups[split]) < 80:
            raise ValueError('V3.7 seeds must be >=80; 0..79 already used/reserved in V3.3–V3.6')
    return groups
