"""Independent math/data tests. MetaDrive not required."""
from __future__ import annotations
from probe_idm_action_v26 import check_recording, summarize_action_subset


def main():
    assert check_recording(4, 5, [0.3, -0.2]) == [0.3, -0.2]
    try:
        check_recording(4, 4, [0.3, -0.2])
        raise AssertionError('Did not reject stale teacher action')
    except RuntimeError as e:
        assert 'not recorded once' in str(e)
    try:
        check_recording(4, 5, [float('nan'), 0.1])
        raise AssertionError('Did not reject nonfinite teacher action')
    except RuntimeError as e:
        assert 'Non-finite' in str(e)
    rows = [
        {'usable': True, 'teacher_action': [-0.2, 0.6], 'model_sign_only_action': [-0.1, 0.3]},
        {'usable': True, 'teacher_action': [0.4, -0.2], 'model_sign_only_action': [0.2, 0.0]},
        {'usable': False, 'teacher_action': [0.8, -0.9], 'model_sign_only_action': [0.8, 0.0]},
    ]
    summary = summarize_action_subset(rows)
    assert summary['n'] == 2
    assert abs(summary['gain_least_squares_through_zero_DIAGNOSTIC_ONLY'] - 2.0) < 1e-8
    assert summary['informative_steer_sign_agreement'] == 1.0
    assert summary['teacher_brake_samples'] == 1
    assert summarize_action_subset([])['n'] == 0
    print('Agent4 V2.6 time-alignment/action-metric unit tests: PASS')


if __name__ == '__main__':
    main()
