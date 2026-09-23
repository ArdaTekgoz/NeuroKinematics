from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neurokinematics.benchmark.contract import load_frozen, strict_json, validate_record, validate_structure
from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.solvers.dls import DLS, SolverStatus, load_solver_config


@pytest.fixture
def record(robot, real_pose):
    q, p, quat = real_pose
    result = DLS(robot).solve(q, p, quat)
    verdict = CandidateValidator(robot).validate(result, p, quat)
    cfg = load_frozen()['config.json']
    return {
        'query_id': 'f05-main-00000000', 'query_group_id': 'root-f05-main-00000000',
        'query_list_sha256': 'a'*64,
        'solver_config_sha256': hashlib.sha256(Path('experiments/F0-05/solver-config.json').read_bytes()).hexdigest(),
        'dataset_manifest_sha256': cfg['immutable_file_sha256']['experiments/F0-04/dataset-manifest.json'],
        'solver_name': 'DLS', 'solver_version': '0.1.0', 'target_source': 'independent_frozen_fk',
        'subset': 'main', 'start_class': 'local', 'q_current': q.tolist(), 'q_candidate': q.tolist(),
        'target_position_m': p.tolist(), 'target_quaternion_wxyz': quat.tolist(),
        'position_error_m': verdict.position_error_m, 'orientation_error_rad': verdict.orientation_error_rad,
        'orientation_error_deg': verdict.orientation_error_deg, 'profile_a_geometry': True, 'profile_b_geometry': True,
        'joint_limits': 'PASS', 'collision': 'NOT_CHECKED', 'solver_status': 'SUCCESS', 'validation_status': 'SUCCESS',
        'reachability': 'KNOWN_REACHABLE', 'reachability_proof_sha256': None,
        'termination_reason': 'STOP_TOLERANCES_REACHED', 'deadline_profile_ms': 10,
        'profile_a_deadline': True, 'profile_b_deadline': True, 'iterations': 0,
        'iteration_availability': 'AVAILABLE', 'first_profile_a_iteration': 0, 'first_profile_b_iteration': 0,
        'solve_elapsed_ns': 100, 'validation_elapsed_ns': 100, 'total_elapsed_ns': 200,
        'measurement_pass_index': 0, 'thermal_state': 'warm',
        'seeds': {k: cfg[k] for k in ('query_seed', 'local_start_seed', 'wide_start_seed', 'boundary_query_seed', 'singularity_query_seed')},
        'frame': 'base_link', 'tcp': 'tool0', 'quaternion_order': 'wxyz',
        'joint_order': [f'joint_{i}' for i in range(1, 7)], 'input_error': None,
    }


def hashes(record):
    return {k: record[k] for k in ('query_list_sha256', 'solver_config_sha256', 'dataset_manifest_sha256')}


def test_record_schema_and_independent_validation(record, robot):
    validate_record(record, validator=CandidateValidator(robot), expected_hashes=hashes(record))


@pytest.mark.parametrize('field,value', [
    ('solver_status', 'CONVERGED'), ('collision', 'PASS'), ('frame', 'world'), ('tcp', 'flange'),
    ('quaternion_order', 'xyzw'), ('joint_order', [f'joint_{i}' for i in range(6, 0, -1)]),
    ('q_candidate', [0]*5), ('iterations', True), ('iterations', -1), ('iterations', 201),
    ('iterations', None), ('iteration_availability', 'NOT_AVAILABLE'), ('measurement_pass_index', 5),
    ('position_error_m', np.nan), ('orientation_error_rad', np.inf), ('total_elapsed_ns', -1),
    ('position_error_m', None), ('position_error_m', 1.), ('profile_a_geometry', False),
    ('validation_status', 'UNRESOLVED'), ('first_profile_a_iteration', 1),
    ('first_profile_a_iteration', None), ('first_profile_b_iteration', None),
    ('query_list_sha256', 'b'*64), ('solver_config_sha256', 'c'*64), ('dataset_manifest_sha256', 'd'*64),
    ('reachability', 'PROVEN_UNREACHABLE'), ('reachability', 'UNRESOLVED'),
    ('reachability_proof_sha256', 'e'*64), ('input_error', 'unexpected'),
    ('q_current', [999]*6), ('target_quaternion_wxyz', [0]*4), ('q_candidate', [999]*6)])
def test_record_faults_rejected(record, robot, field, value):
    expected = hashes(record)
    record[field] = value
    with pytest.raises(ValueError):
        validate_record(record, validator=CandidateValidator(robot), expected_hashes=expected)


def test_missing_and_extra_field_rejected(record, robot):
    expected = hashes(record)
    for mode in ('missing', 'extra'):
        changed = deepcopy(record)
        if mode == 'missing': del changed['iterations']
        else: changed['q_target'] = [0]*6
        with pytest.raises(ValueError):
            validate_record(changed, validator=CandidateValidator(robot), expected_hashes=expected)


@pytest.mark.parametrize('payload', ['{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
                                    '{"x":1e999}', '{"x":1,"x":2}', '{"x":'])
def test_strict_json(payload):
    with pytest.raises(ValueError): strict_json(payload)


def test_timeout_and_late_validator_cannot_pass(record, robot):
    expected = hashes(record)
    record['solve_elapsed_ns'] = 9_000_000
    record['validation_elapsed_ns'] = 2_000_000
    record['total_elapsed_ns'] = 11_000_000
    with pytest.raises(ValueError, match='deadline'):
        validate_record(record, validator=CandidateValidator(robot), expected_hashes=expected)
    record['profile_a_deadline'] = record['profile_b_deadline'] = False
    validate_record(record, validator=CandidateValidator(robot), expected_hashes=expected)
    record.update(solver_status='TIMEOUT', validation_status='TIMEOUT', total_elapsed_ns=200,
                  solve_elapsed_ns=100, validation_elapsed_ns=100)
    validate_record(record, validator=CandidateValidator(robot), expected_hashes=expected)
    record['profile_b_deadline'] = True
    with pytest.raises(ValueError, match='deadline'):
        validate_record(record, validator=CandidateValidator(robot), expected_hashes=expected)


def test_invalid_input_preserves_nulls(record, robot):
    expected = hashes(record)
    record.update(target_source='invalid_fixture', subset='invalid', start_class='fixture', q_current=None,
                  q_candidate=None, target_quaternion_wxyz=None, solver_status='INVALID_INPUT',
                  validation_status='INVALID_INPUT', reachability='UNRESOLVED', input_error='zero quaternion',
                  iterations=None, iteration_availability='NOT_AVAILABLE', first_profile_a_iteration=None,
                  first_profile_b_iteration=None, position_error_m=None, orientation_error_rad=None,
                  orientation_error_deg=None, joint_limits='NOT_AVAILABLE', profile_a_geometry=False,
                  profile_b_geometry=False, profile_a_deadline=False, profile_b_deadline=False)
    validate_record(record, validator=CandidateValidator(robot), expected_hashes=expected)
    record['iterations'] = 0
    record['iteration_availability'] = 'AVAILABLE'
    with pytest.raises(ValueError, match='unavailable iteration'):
        validate_record(record, validator=CandidateValidator(robot), expected_hashes=expected)


def test_frozen_inputs_and_statuses():
    files = load_frozen()
    cfg, contract = files['config.json'], files['benchmark-contract.json']
    assert cfg['query_counts'] == {'main': 10000, 'boundary': 1000, 'singularity': 1000}
    assert cfg['local_fraction'] == .5 and cfg['measurement_passes'] == 5
    assert cfg['deadline_profiles_ms'] == [10, 50]
    assert [cfg[k] for k in ('query_seed','local_start_seed','wide_start_seed','boundary_query_seed','singularity_query_seed')] == list(range(20260925, 20260930))
    assert set(contract['status_model']['terminal_statuses']) == {s.value for s in SolverStatus}
    for name, value in cfg['immutable_file_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == value
    for key in ('solver_status', 'validation_status'):
        assert set(files['benchmark-schema.json']['properties'][key]['enum']) == {s.value for s in SolverStatus}
    solver = load_solver_config()
    assert solver.stop_position_m <= cfg['profiles']['B']['position_m']
    assert solver.stop_orientation_rad <= cfg['profiles']['B']['orientation_rad']
    assert solver.characteristic_length_m == cfg['characteristic_length_m']
    assert contract['measurement']['expected_measurement_records'] == sum(cfg['query_counts'].values())*5*2


@pytest.mark.parametrize('name', ['config.json', 'solver-config.json', 'benchmark-contract.json', 'benchmark-schema.json'])
def test_frozen_byte_mutation_rejected(tmp_path, name):
    source = Path('experiments/F0-05')
    target = tmp_path/source
    target.mkdir(parents=True)
    for path in [*source.glob('*.json')]:
        if path.name in {name, 'config.json', 'solver-config.json', 'benchmark-contract.json', 'benchmark-schema.json', 'stage1-frozen-hashes.json'}:
            (target/path.name).write_bytes(path.read_bytes()+(b' ' if path.name == name else b''))
    with pytest.raises(ValueError, match='frozen artifact hash mismatch'):
        load_frozen(tmp_path)


def test_solver_parameter_mutation_rejected(tmp_path):
    config = json.loads(Path('experiments/F0-05/solver-config.json').read_bytes())
    config['damping'] = 0
    path = tmp_path/'solver.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='frozen solver config'):
        load_solver_config(path)


def test_schema_validator_rejects_unknown_keyword():
    with pytest.raises(ValueError, match='unsupported schema'):
        validate_structure(1, {'type': 'number', 'unknown': True})
