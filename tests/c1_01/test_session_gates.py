"""Runtime drift and evidence substitution must prevent the next main stage."""

from collections import Counter
import importlib.util
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from neurokinematics.core.contract import ROOT, load_contract, sha256
from neurokinematics.core.runner import FROZEN_QUERY_PATH, load_queries


SCRIPT = Path(os.environ.get("C101_SESSION_SCRIPT", ROOT / "scripts/c101_session.py"))
SPEC = importlib.util.spec_from_file_location("c101_session_under_test", SCRIPT)
SESSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SESSION)
RUNTIME = "a" * 64


@pytest.mark.parametrize('limit,swap', [('8589934592', '0'), ('max', '0'), ('4294967296', '0'), ('8589934592', 'max')])
def test_memory_ceiling_rejects_unlimited_changed_or_swap(tmp_path, limit, swap):
    (tmp_path / 'memory.max').write_text(limit)
    (tmp_path / 'memory.swap.max').write_text(swap)
    if (limit, swap) == ('8589934592', '0'):
        assert SESSION.memory_policy(tmp_path)['memory_max_bytes'] == 8589934592
    else:
        with pytest.raises(ValueError, match='ceiling'):
            SESSION.memory_policy(tmp_path)


def test_memory_ceiling_requires_both_controller_files(tmp_path):
    (tmp_path / 'memory.max').write_text('8589934592')
    with pytest.raises(ValueError, match='cgroup v2'):
        SESSION.memory_policy(tmp_path)


def test_host_memory_drift_is_recorded_without_changing_resource_identity(tmp_path, monkeypatch):
    runtime = {'memory_policy': {'controller': 'cgroup-v2', 'memory_max_bytes': 8589934592, 'swap_max_bytes': 0}}
    session = tmp_path / 'session'
    session.mkdir()
    snapshots = iter([{'mem_total': '11886228 kB', 'mem_available': '9000000 kB'},
                      {'mem_total': '11886236 kB', 'mem_available': '8000000 kB'}])
    monkeypatch.setattr(SESSION, 'memory_observation', lambda: next(snapshots))
    monkeypatch.setattr(SESSION, 'capture_runtime', lambda *_: runtime.copy())
    def fake_prepare(path, runtime_sha):
        (path / 'prepare').mkdir()
        return {'stage': 'prepare', 'status': 'PASS', 'runtime_sha256': runtime_sha}
    monkeypatch.setattr(SESSION, 'preparation', fake_prepare)
    monkeypatch.setattr(SESSION.sys, 'argv', ['c101_session.py', 'prepare', '--session', str(session), '--image-id', 'sha256:test'])
    assert SESSION.main() == 0
    assert SESSION.read_json(session / 'runtime-lock.json') == runtime
    gate = SESSION.read_json(session / 'prepare/gate.json')
    assert gate['host_memory_observations']['start']['mem_total'] == '11886228 kB'
    assert gate['host_memory_observations']['end']['mem_total'] == '11886236 kB'
    assert not (tmp_path / '.c101-active').exists()


@pytest.mark.parametrize("change", ["system", "machine", "release", "affinity", "transport", "threads"])
def test_policy_rejects_drift(change):
    args = {"system": "Linux", "machine": "x86_64", "release": {"ID": "ubuntu", "VERSION_ID": "24.04"},
            "affinity": [0, 1], "environment": dict(SESSION.ENVIRONMENT)}
    SESSION.require_policy(**args)
    if change in ("system", "machine"):
        args[change] = "other"
    elif change == "release":
        args[change]["VERSION_ID"] = "22.04"
    elif change == "affinity":
        args[change] = [2, 3]
    elif change == "transport":
        args["environment"]["FASTDDS_BUILTIN_TRANSPORTS"] = "DEFAULT"
    else:
        args["environment"]["OMP_NUM_THREADS"] = "2"
    with pytest.raises(ValueError):
        SESSION.require_policy(**args)


def test_pilot_uses_first_two_frozen_queries_of_each_group():
    rows, _ = load_queries(FROZEN_QUERY_PATH, load_contract())
    selected = SESSION.pilot_selection(rows)
    groups = Counter((row["subset"], row["start_class"]) for row in selected)
    assert len(selected) == 12 and len(groups) == 6
    assert set(groups.values()) == {2}
    assert selected == sorted(selected, key=lambda row: rows.index(row))
    for group in groups:
        expected = [row for row in rows if (row["subset"], row["start_class"]) == group][:2]
        assert [row for row in selected if (row["subset"], row["start_class"]) == group] == expected
    with pytest.raises(ValueError, match="six groups"):
        SESSION.pilot_selection([row for row in rows if row["subset"] != "boundary"])


def junit(path, names, *, failures=0, errors=0, skipped=0):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", failures=str(failures), errors=str(errors), skipped=str(skipped))
    for name in names:
        ET.SubElement(suite, "testcase", classname="tests.c1_01.test_gate", name=name)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


@pytest.mark.parametrize("mutation", ["none", "different", "missing", "duplicate", "failure", "error", "skip"])
def test_junit_requires_exact_passing_identities(tmp_path, mutation):
    path = tmp_path / "result.xml"
    names = ["test_one", "test_two[param]"]
    counts = {}
    if mutation == "different":
        names[1] = "test_unrelated[param]"
    elif mutation == "missing":
        names.pop()
    elif mutation == "duplicate":
        names[1] = names[0]
    elif mutation in ("failure", "error", "skip"):
        counts[{"failure": "failures", "error": "errors", "skip": "skipped"}[mutation]] = 1
    junit(path, names, **counts)
    nodeids = ["tests/c1_01/test_gate.py::test_one", "tests/c1_01/test_gate.py::test_two[param]"]
    if mutation == "none":
        assert SESSION.xml_result(path, nodeids) == 2
    else:
        with pytest.raises(ValueError):
            SESSION.xml_result(path, nodeids)


def add_gate(session, stage):
    output = session / stage
    output.mkdir()
    evidence = output / "evidence.txt"
    evidence.write_text(stage, encoding="utf-8")
    gate = {"stage": stage, "status": "MEASURED_UNVERIFIED" if stage == "full" else "PASS",
            "runtime_sha256": RUNTIME, "files": {"evidence.txt": sha256(evidence)}}
    if stage in ("smoke", "pilot", "full"):
        gate["prepare_gate_sha256"] = sha256(session / "prepare/gate.json")
        predecessor = {"smoke": None, "pilot": "smoke", "full": "pilot"}[stage]
        gate["previous_gate_sha256"] = sha256(session / predecessor / "gate.json") if predecessor else None
    elif stage == "verify":
        gate["full_gate_sha256"] = sha256(session / "full/gate.json")
    SESSION.write_json(output / "gate.json", gate)


@pytest.fixture
def chain(tmp_path):
    for stage in ("prepare", "smoke", "pilot", "full", "verify"):
        add_gate(tmp_path, stage)
    return tmp_path


@pytest.mark.parametrize("mutation", ["none", "evidence", "missing", "runtime", "escape"])
def test_gate_rejects_changed_or_unbound_evidence(chain, mutation):
    path = chain / "prepare/gate.json"
    gate = SESSION.read_json(path)
    if mutation == "evidence":
        (path.parent / "evidence.txt").write_text("changed", encoding="utf-8")
    elif mutation == "missing":
        (path.parent / "evidence.txt").unlink()
    elif mutation == "runtime":
        gate["runtime_sha256"] = "b" * 64
    elif mutation == "escape":
        outside = chain / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        gate["files"] = {"../outside.txt": sha256(outside)}
    SESSION.write_json(path, gate)
    if mutation == "none":
        assert SESSION.check_gate(chain, "prepare", RUNTIME)["status"] == "PASS"
    else:
        with pytest.raises(ValueError):
            SESSION.check_gate(chain, "prepare", RUNTIME)


@pytest.mark.parametrize("replaced_stage", ["prepare", "smoke", "pilot", "full"])
def test_verified_gate_rejects_replaced_predecessor(chain, replaced_stage):
    assert SESSION.check_gate(chain, "verify", RUNTIME)["status"] == "PASS"
    path = chain / replaced_stage / "gate.json"
    gate = SESSION.read_json(path)
    evidence = path.parent / "evidence.txt"
    evidence.write_text("replacement with an internally consistent file hash", encoding="utf-8")
    gate["files"]["evidence.txt"] = sha256(evidence)
    SESSION.write_json(path, gate)
    with pytest.raises(ValueError):
        SESSION.check_gate(chain, "verify", RUNTIME)
