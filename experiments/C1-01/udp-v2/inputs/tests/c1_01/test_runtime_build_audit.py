"""A local worker rebuild must not silently replace its dependency closure."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def auditor(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts/audit_c101_runtime_build.py"
    spec = importlib.util.spec_from_file_location("c101_runtime_build_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    responses = {
        ("dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Architecture}\n"): "example\t1.2\tamd64",
        ("pixi", "--version"): "pixi 0.81.0",
        ("git", "-C", "/opt/c101/external/pick_ik", "rev-parse", "HEAD"): "a" * 40,
        ("git", "-C", "/opt/c101/external/pick_ik", "status", "--porcelain"): "",
    }
    # Use POSIX paths under both Windows and Linux for this isolated command fake.
    def command(*args):
        return responses[tuple(arg.replace("\\", "/") for arg in args)]
    monkeypatch.setattr(module, "command", command)
    monkeypatch.setattr(module, "digest", lambda path: "lock-hash")
    base = {
        "dpkg_closure_sha256": hashlib.sha256(b"example\t1.2\tamd64").hexdigest(),
        "dpkg_package_count": 1, "pixi_lock_sha256": "lock-hash", "pixi": "pixi 0.81.0",
        "sources": {"pick_ik": {"commit": "a" * 40, "url": "https://example.invalid/upstream"}},
    }
    return module, base, responses


def test_same_dependency_closure_is_accepted(auditor):
    module, base, _ = auditor
    assert module.audit(base, recorded=copy.deepcopy(base))["status"] == "PASS"


@pytest.mark.parametrize("field", ["dpkg_closure_sha256", "dpkg_package_count", "pixi_lock_sha256", "pixi"])
def test_changed_inherited_dependency_is_rejected(auditor, field):
    module, base, _ = auditor
    base[field] = "changed"
    with pytest.raises(ValueError, match="inherited dependency changed"):
        module.audit(base)


@pytest.mark.parametrize("revision,dirty", [("b" * 40, ""), ("a" * 40, " M source.cpp")])
def test_external_source_drift_is_rejected(auditor, revision, dirty):
    module, base, responses = auditor
    responses[("git", "-C", "/opt/c101/external/pick_ik", "rev-parse", "HEAD")] = revision
    responses[("git", "-C", "/opt/c101/external/pick_ik", "status", "--porcelain")] = dirty
    with pytest.raises(ValueError, match="inherited external source"):
        module.audit(base)


def test_new_lock_cannot_claim_a_different_dependency_closure(auditor):
    module, base, _ = auditor
    recorded = copy.deepcopy(base)
    recorded["dpkg_closure_sha256"] = "replacement"
    with pytest.raises(ValueError, match="recorded dependency differs"):
        module.audit(base, recorded=recorded)
