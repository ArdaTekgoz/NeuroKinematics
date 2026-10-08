"""Synthetic negative controls only: never open benchmark/test shards."""
import copy
import json
import numpy as np
import pytest
from neurokinematics.neural.c106 import Validator, features, timed_query, paired_bootstrap, h2_decision
from neurokinematics.data.factory import canonical_quaternion
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK


@pytest.fixture
def example():
    validator = Validator()
    q = (validator.lower + validator.upper) / 2
    pose = PinocchioFK(validator.robot).reference_forward_kinematics(q)
    query = dict(query_id='synthetic', target_position_m=pose[:3, 3].tolist(),
                 target_quaternion_wxyz=canonical_quaternion(pose[:3, :3]).tolist(), q_current=q.tolist())
    norm = dict(position_mean_m=[0, 0, 0], position_std_m=[1, 1, 1])
    return validator, q, query, norm


def test_independent_fk_and_timing(example):
    validator, q, query, norm = example
    result = timed_query(query, lambda x: q, norm, validator)
    assert result['profile_a'] and result['profile_b']
    assert result['position_error_m'] < 1e-9
    assert result['elapsed_ns'] > 0 and result['collision'] == 'NOT_CHECKED'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('kind', ['nan', 'inf', 'shape', 'limit', 'position', 'orientation'])
def test_negative_geometry(example, kind):
    validator, q, query, _ = example
    if kind == 'nan': q[0] = np.nan
    if kind == 'inf': q[0] = np.inf
    if kind == 'shape': q = q[:5]
    if kind == 'limit': q[0] = validator.upper[0] + 1e-8
    if kind == 'position': query['target_position_m'][0] += .01
    if kind == 'orientation': query['target_quaternion_wxyz'] = [1, 0, 0, 0]
    before = q.copy()
    result = validator.check(q, query)
    assert not result['profile_a'] and not result['profile_b']
    assert np.array_equal(before, q, equal_nan=True)  # no silent clamp


def test_input_whitelist(example):
    validator, _, query, norm = example
    expected = features(query, norm, validator.robot.limits)
    query.update(q_target=[999] * 6, teacher_solution=[np.nan] * 6, label_present=False)
    assert np.array_equal(features(query, norm, validator.robot.limits), expected)
    assert expected.shape == (13,)


@pytest.mark.parametrize('field,value', [('q_current', [float('nan')]*6), ('q_current', [999]*6),
                                       ('target_quaternion_wxyz', [0]*4), ('target_position_m', [0]*2)])
def test_bad_input(example, field, value):
    validator, _, query, norm = example
    query[field] = value
    with pytest.raises(ValueError): features(query, norm, validator.robot.limits)


def rows(success=False):
    return [dict(query_id=f'{subset}-{i}', group_id=f'{subset}-root-{i//2}',
                 subsets=[subset], seed=s, success=success)
            for s in (1, 2, 3) for subset in ('main', 'boundary', 'singularity') for i in range(8)]


def test_zero_effect_and_denominator():
    result = paired_bootstrap(rows(), rows(), seeds=(1, 2, 3), repeats=100)
    assert result['unique_queries'] == 24 and result['root_groups'] == 12
    assert result['mean_over_fixed_seeds']['main']['ci95'] == [0, 0]
    assert h2_decision(result) == 'REJECTED'
    assert h2_decision(result, integrity=False) == 'INCONCLUSIVE_TECHNICAL'


def test_paired_effect_sign_reproducibility_and_seed_axis():
    a, b = rows(True), rows(False)
    for row in a:
        row['success'] = row['seed'] != 2 and int(row['query_id'].split('-')[-1]) % 2 == 0
    result = paired_bootstrap(a, b, seeds=(1, 2, 3), repeats=300)
    assert result == paired_bootstrap(a, b, seeds=(1, 2, 3), repeats=300)
    assert result['mean_over_fixed_seeds']['main']['difference'] == pytest.approx(1/3)
    assert result['seed_range']['main'] == [0, .5]
    assert h2_decision(result) == 'SUPPORTED'
    inverse = paired_bootstrap(b, a, seeds=(1, 2, 3), repeats=300)
    assert inverse['mean_over_fixed_seeds']['main']['difference'] == pytest.approx(-1/3)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'group', 'subset', 'seed', 'binary'])
def test_pair_integrity(fault):
    a, b = rows(), rows()
    if fault == 'missing': b.pop()
    if fault == 'duplicate': b.append(b[0])
    if fault == 'group': b[0]['group_id'] = 'wrong'
    if fault == 'subset': b[0]['subsets'] = ['boundary']
    if fault == 'seed': b[0]['seed'] = 5
    if fault == 'binary': b[0]['success'] = 1
    with pytest.raises(ValueError): paired_bootstrap(a, b, seeds=(1, 2, 3), repeats=10)


def test_overlap_is_single_root_single_denominator():
    a = rows()
    for row in a:
        if row['query_id'].startswith('boundary'):
            row['subsets'] = ['boundary', 'singularity']
    result = paired_bootstrap(a, copy.deepcopy(a), seeds=(1, 2, 3), repeats=30)
    assert result['unique_queries'] == 24 and result['root_groups'] == 12
    assert result['subset_n'] == dict(main=8, boundary=8, singularity=16)


@pytest.mark.parametrize('hard,main,decision', [([.02,.03],[-.01,0],'SUPPORTED'),
    ([.01,.03],[-.01,0],'INCONCLUSIVE'), ([0,.019],[-.01,0],'REJECTED'),
    ([.02,.03],[-.03,-.011],'REJECTED'), ([.02,.03],[-.02,0],'INCONCLUSIVE')])
def test_decision_thresholds(hard, main, decision):
    result = {'mean_over_fixed_seeds': {'hard_equal_weight':dict(difference=np.mean(hard),ci95=hard),
                                      'main':dict(difference=np.mean(main),ci95=main)}}
    assert h2_decision(result) == decision


def test_hard_subsets_equal_weight_not_pooled():
    a, b = rows(), rows()
    # Boundary has 8 queries, singularity has 16; hard must remain 50/50.
    for collection, success in ((a, True), (b, False)):
        for row in collection:
            row['success'] = success and row['subsets'] == ['boundary']
        collection.extend(dict(query_id=f'extra-{i}', group_id=f'extra-{i}',
                               subsets=['singularity'], seed=s, success=False)
                          for s in (1, 2, 3) for i in range(8))
    result = paired_bootstrap(a, b, seeds=(1, 2, 3), repeats=30)
    assert result['mean_over_fixed_seeds']['hard_equal_weight']['difference'] == .5


@pytest.mark.parametrize('fault', ['hash', 'bytes', 'rows', 'escape'])
def test_frozen_input_negative_controls(tmp_path, monkeypatch, fault):
    from pathlib import Path
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts'))
    checker = importlib.import_module('check_c106_stage1')
    monkeypatch.setattr(checker, 'ROOT', tmp_path)
    target = tmp_path / 'fixture'
    target.write_bytes(b'one\ntwo\n')
    row = dict(path='fixture', **checker.fingerprint(target, True))
    checker.verify_row(row)
    if fault == 'hash': row['sha256'] = '0'*64
    if fault == 'bytes': row['bytes'] += 1
    if fault == 'rows': row['rows'] += 1
    if fault == 'escape': row['path'] = '../outside'
    with pytest.raises(ValueError): checker.verify_row(row)
