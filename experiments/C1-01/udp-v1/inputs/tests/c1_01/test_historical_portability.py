"""Supplement historical Windows fixture with a native-runtime mutation test.

The original F0-06 test and archived evidence remain unchanged.
"""
import copy
import json

import numpy as np
import pytest

from neurokinematics.core.contract import ROOT
from neurokinematics.benchmark.contract import validate_record
from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.solvers.dls import SolverResult, SolverStatus


def test_native_runtime_historical_schema_and_verdict_mutations():
    archived = json.loads((ROOT / 'experiments/F0-06/reproduction/a/results.jsonl')
                          .read_bytes().splitlines()[0])
    validator = CandidateValidator()
    result = SolverResult(SolverStatus(archived['solver_status']), archived['termination_reason'],
                          np.asarray(archived['q_candidate']), archived['iterations'],
                          archived['first_profile_a_iteration'], archived['first_profile_b_iteration'],
                          archived['solve_elapsed_ns'])
    verdict = validator.validate(result, archived['target_position_m'], archived['target_quaternion_wxyz'])
    # Preserve categorical acceptance decisions; no tolerance or evidence edits.
    for key in ('profile_a_geometry', 'profile_b_geometry', 'joint_limits', 'collision'):
        assert archived[key] == getattr(verdict, key)
    assert archived['validation_status'] == verdict.status
    native = copy.deepcopy(archived)
    for key in ('position_error_m', 'orientation_error_rad', 'orientation_error_deg'):
        native[key] = getattr(verdict, key)
    hashes = {key: archived[key] for key in
              ('query_list_sha256', 'solver_config_sha256', 'dataset_manifest_sha256')}
    validate_record(native, validator=validator, expected_hashes=hashes)
    missing = copy.deepcopy(native)
    del missing['solver_status']
    with pytest.raises(ValueError):
        validate_record(missing, validator=validator, expected_hashes=hashes)
    corrupt = copy.deepcopy(native)
    corrupt['orientation_error_deg'] += 1e-6
    with pytest.raises(ValueError, match='independent verdict mismatch'):
        validate_record(corrupt, validator=validator, expected_hashes=hashes)
