"""No slow marker: normal test-f03 always executes all 256 q and all h."""

import pytest

from neurokinematics.kinematics.jacobian_validation import run_validation


@pytest.fixture(scope='module')
def evidence(robot):
    return run_validation(robot)


@pytest.mark.parametrize('h', ['1e-05', '1e-06', '1e-07'])
def test_tf03_256_all_pairs_and_handpicked(evidence, h, record_property):
    summary, diagnostics, selected = evidence
    assert summary['sample_count'] == 256 and summary['sample_shape'] == [256, 6]
    assert summary['seed'] == 20260919 and summary['dtype'] == 'float64'
    assert summary['sample_sha256'] == '678eb4286863026880792ef0cc3c0a9d4f92e16f85b1aa009705cbf0b59b26e7'
    record_property('sample_sha256', summary['sample_sha256'])
    record_property('h_rad', h)
    run = summary['runs'][h]
    for group in (run, run['handpicked']):
        for key, pair in group['pairs'].items():
            assert pair['valid_count'] == pair['N']
            assert pair['normalized_error']['max'] <= 1e-5
            assert pair['invalid_result_count'] == pair['nonfinite_result_count'] == pair['threshold_exceeded_count'] == 0
            record_property(key + ('_handpicked' if group is not run else ''), pair['normalized_error']['max'])
    assert len(selected[h]) == 21
    assert len(diagnostics[h]['singular_subgroup']) == 3
    wrist = next(r for r in selected[h] if r['name'] == 'wrist_aligned')
    assert wrist['metrics']['geometric']['sigma_min'] < 1e-12
    assert run['low_sigma_sample']['sample_index'] in range(256)
    assert summary['status'] == 'PASS'
