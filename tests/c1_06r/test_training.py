"""Real training interruption equivalence and fail-closed identity controls."""
import copy
from pathlib import Path
import pytest
import torch
from neurokinematics.neural import c106r, c106r_training as training
from neurokinematics.neural.c104 import load_data, read_json


@pytest.fixture(scope="module")
def small():
    c106r.configure()
    train, validation = load_data(label_fk=False)
    return c106r.subsets(train, read_json(c106r.CONFIG))["local64"], validation.take(__import__('numpy').arange(32))


@pytest.mark.parametrize("arm", ["Q", "FK", "Q_TANH", "FK_TANH"])
def test_real_cuda_resume_matches_uninterrupted(small, tmp_path, arm):
    assert torch.cuda.is_available(), "GPU handoff tests must actually execute CUDA"
    spec = copy.deepcopy(read_json(training.PROTOCOL))
    spec.update(epochs=3, batch_size=16, validation_every_epochs=1)
    seed = spec["seed_list"][0]
    contract = dict(test="resume", arm=arm, seed=seed)
    full = training.train_run(*small, spec, arm, seed, tmp_path/"full", contract)
    training.train_run(*small, spec, arm, seed, tmp_path/"resume", contract, stop_after=1)
    resumed = training.train_run(*small, spec, arm, seed, tmp_path/"resume", contract, resume=True)
    assert full["final_state_hash"] == resumed["final_state_hash"]
    assert full["best"] == resumed["best"]
    assert [h["loss"] for h in full["history"]] == [h["loss"] for h in resumed["history"]]
    assert [h["permutation_sha256"] for h in full["history"]] == [h["permutation_sha256"] for h in resumed["history"]]


def test_checkpoint_corruption_rejected(tmp_path):
    contract = dict(task="test")
    training.save_state(tmp_path, dict(epoch=1, contract=contract))
    pointer = read_json(tmp_path/"checkpoint.json")
    (tmp_path/pointer["file"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        training.load_state(tmp_path, contract)


def test_resume_config_drift_rejected(tmp_path):
    training.save_state(tmp_path, dict(epoch=1, contract={"seed": 1}))
    with pytest.raises(ValueError, match="contract"):
        training.load_state(tmp_path, {"seed": 2})


def test_unpublished_slot_cannot_replace_committed_checkpoint(tmp_path):
    contract = dict(task="test")
    training.save_state(tmp_path, dict(epoch=1, contract=contract))
    (tmp_path/"last-1.pt").write_bytes(b"incomplete next checkpoint")
    assert training.load_state(tmp_path, contract)["epoch"] == 1


def test_selection_prioritizes_pose_success_over_q_label_loss():
    def metric(success, p, invalid=False):
        return dict(n=1, profile_a=success, nonfinite=0, out_of_limits=int(invalid),
                    rows=[dict(valid=not invalid, position_m=p, orientation_deg=0)])
    assert training.checkpoint_rank(metric(1, .001)) < training.checkpoint_rank(metric(0, .003))
    assert training.checkpoint_rank(metric(0, .003)) < training.checkpoint_rank(metric(0, None, True))


def test_unregistered_arm_cannot_fall_through():
    with pytest.raises(ValueError, match="unknown"):
        training.normalized(torch.zeros(6), "typo")
