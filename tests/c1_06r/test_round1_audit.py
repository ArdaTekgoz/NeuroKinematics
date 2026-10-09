"""Acceptance audit rejects missing, modified and incomplete evidence."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
import pickle

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("round1_audit", ROOT / "scripts/audit_c106r_round1.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_historical_torch_version_metadata_requires_scoped_allowlist(tmp_path):
    contract = dict(torch=torch.__version__)
    audit.training.save_state(tmp_path, dict(epoch=2000, contract=contract))
    with pytest.raises(pickle.UnpicklingError):
        audit.training.load_state(tmp_path, contract)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        assert audit.training.load_state(tmp_path, contract)["epoch"] == 2000
    # Reading an old file does not weaken later checkpoint loads globally.
    with pytest.raises(pickle.UnpicklingError):
        audit.training.load_state(tmp_path, contract)


@pytest.mark.parametrize("mutation", ["modified", "missing", "extra", "duplicate", "escape"])
def test_manifest_rejects_corruption(tmp_path, mutation):
    (tmp_path / "record.json").write_text('{}')
    line = audit.sha(tmp_path / "record.json") + "  record.json\n"
    (tmp_path / "SHA256SUMS").write_text(line)
    assert len(audit.verify_manifest(tmp_path)) == 1
    if mutation == "modified":
        (tmp_path / "record.json").write_text('{"changed":true}')
    elif mutation == "missing":
        (tmp_path / "record.json").unlink()
    elif mutation == "extra":
        (tmp_path / "extra.json").write_text('{}')
    elif mutation == "duplicate":
        (tmp_path / "SHA256SUMS").write_text(line + line)
    else:
        (tmp_path / "SHA256SUMS").write_text('0' * 64 + '  ../outside.json\n')
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.verify_manifest(tmp_path)


@pytest.mark.parametrize("mutation", ["missing_epoch", "duplicate_epoch", "updates", "samples", "cadence", "denominator", "nan_loss"])
def test_history_rejects_incomplete_training(mutation):
    config = dict(epochs=2, validation_every_epochs=1, budgets=dict(updates_per_epoch=60))
    history = [dict(epoch=e, updates=60, samples=15204, loss=dict(q=.1),
                    validation=dict(n=3600), selection_rank=[0, 0, 1.]) for e in (1, 2)]
    assert audit.verify_history(history, config)["epoch"] == 1  # earliest exact tie
    bad = copy.deepcopy(history)
    if mutation == "missing_epoch":
        bad.pop()
    elif mutation == "duplicate_epoch":
        bad[1]["epoch"] = 1
    elif mutation in ("updates", "samples"):
        bad[0][mutation] -= 1
    elif mutation == "cadence":
        del bad[0]["validation"]
    elif mutation == "denominator":
        bad[0]["validation"]["n"] = 3599
    else:
        bad[0]["loss"]["q"] = float('nan')
    with pytest.raises(ValueError):
        audit.verify_history(bad, config)
