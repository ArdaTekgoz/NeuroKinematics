import hashlib
import json

import numpy as np
import pytest

from neurokinematics.kinematics.model import ROOT, FROZEN_HASHES, load_robot
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.pinocchio_jacobian import PinocchioJacobian
from neurokinematics.kinematics.finite_difference import CentralDifference
from neurokinematics.kinematics.jacobian_validation import (
    load_config, sample_configurations, sample_hash, evaluate, json_safe, characteristic_length,
    handpicked_configurations, inspect,
)
from neurokinematics.kinematics.validation import write_json


def test_sample_determinism_margin_hash(robot):
    a, b = sample_configurations(robot), sample_configurations(robot)
    assert a.shape == (256, 6) and a.dtype == np.float64
    assert np.array_equal(a, b)
    assert sample_hash(a) == hashlib.sha256(a.astype('<f8').tobytes(order='C')).hexdigest()
    assert sample_hash(a) == '678eb4286863026880792ef0cc3c0a9d4f92e16f85b1aa009705cbf0b59b26e7'
    assert sample_hash(a) == sample_hash(np.asfortranarray(a)) == sample_hash(a.astype('>f8'))
    assert np.all(a-1e-5 >= np.array(robot.limits)[:, 0])
    assert np.all(a+1e-5 <= np.array(robot.limits)[:, 1])


@pytest.mark.parametrize("key,value", [("sample_count", 100), ("seed", 1), ("primary_h_rad", 1e-5),
                                        ("sensitivity_h_rad", []), ("normalized_error_threshold", 1e-4),
                                        ("limit_margin_rad", 0), ("characteristic_length_m", 1.)])
def test_frozen_config(tmp_path, key, value):
    config = load_config(); config[key] = value
    path = tmp_path / 'config.json'; write_json(path, config)
    with pytest.raises(ValueError, match="frozen T-F03"):
        load_config(path)


@pytest.mark.parametrize("path", list(FROZEN_HASHES))
def test_immutable_hash_rejection(tmp_path, path):
    for p in FROZEN_HASHES:
        destination = tmp_path / p; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/p).read_bytes() + (b' ' if p == path else b''))
    with pytest.raises(ValueError, match="immutable hash mismatch"):
        load_robot(tmp_path)


@pytest.mark.parametrize("fault", ["nan", "inf", "shape", "float32", "exception", "wrong", "near"])
def test_failure_evidence_preserves_every_sample(robot, tmp_path, fault):
    original = IndependentJacobian(robot).jacobian
    def bad(q):
        j = original(q)
        if fault in ('nan', 'inf'):
            j[0, 0] = float(fault)
        elif fault == 'shape':
            j = j[:3]
        elif fault == 'float32':
            j = j.astype(np.float32)
        elif fault == 'exception':
            raise RuntimeError('injected failure')
        elif fault == 'wrong':
            j[:3] += .01
        else:
            # Exact desired normalized error relative to the unchanged oracle.
            ell = characteristic_length()
            norm = np.linalg.norm(np.vstack((j[:3]/ell, j[3:])))
            j[0, 0] += .95e-5 * max(1., norm) * ell
        return j
    samples = sample_configurations(robot)[:2]
    methods = {'geometric': bad, 'pinocchio': PinocchioJacobian(robot).jacobian,
               'central': CentralDifference(robot).jacobian}
    summary, diagnostics, rows = evaluate(robot, samples, characteristic_length(), 1e-6, methods=methods)
    assert len(rows) == 2 and [r['sample_index'] for r in rows] == [0, 1]
    assert np.array_equal([r['q'] for r in rows], samples)
    assert summary['status'] == ('PASS' if fault == 'near' else 'FAIL')
    if fault in ('wrong', 'near'):
        assert len(diagnostics['failed_or_near_threshold']) == 2
    else:
        assert len(diagnostics['invalid_results']) == 2
        assert summary['pairs']['geometric_vs_pinocchio']['invalid_result_count'] == 2
        assert summary['pairs']['geometric_vs_pinocchio']['nonfinite_result_count'] == (2 if fault in ('nan', 'inf') else 0)
    path = tmp_path/'evidence.json'
    write_json(path, json_safe({'summary': summary, 'diagnostics': diagnostics}))
    assert json.loads(path.read_text())['summary']['status'] == summary['status']


def test_contract_and_f02_api_regression(robot):
    from neurokinematics.kinematics import IndependentFK
    from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
    a, b = IndependentFK(robot), PinocchioFK(robot)
    for q in handpicked_configurations(robot).values():
        assert np.linalg.norm(a.forward_kinematics(q)-b.reference_forward_kinematics(q)) <= 1e-9
    contract = inspect(robot)
    assert contract['row_order'] == ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    assert contract['point'] == 'tool0' and contract['axes'] == 'base_link'
    assert [j['name'] for j in contract['chain']][-2:] == ['joint_6-flange', 'flange-tool0']
    assert characteristic_length() == .9015


@pytest.mark.parametrize("backend", [IndependentJacobian, PinocchioJacobian, CentralDifference])
@pytest.mark.parametrize("q", [[0]*5, [np.nan]*6, [np.inf]*6, [90, 0, 0, 0, 0, 0]])
def test_bad_q(robot, backend, q):
    with pytest.raises(ValueError):
        backend(robot).jacobian(q)


def test_evidence_hash_corruption_is_rejected(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('f03_runner', ROOT/'scripts/run_f03_acceptance.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    payload = tmp_path/'evidence.json'; payload.write_bytes(b'{}\n')
    sums = tmp_path/'SHA256SUMS'
    sums.write_text(hashlib.sha256(payload.read_bytes()).hexdigest()+'  evidence.json\n')
    assert runner.verify_hashes(tmp_path) == 1
    payload.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ValueError, match='evidence hash mismatch'):
        runner.verify_hashes(tmp_path)
