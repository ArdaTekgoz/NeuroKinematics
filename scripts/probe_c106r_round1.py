"""Fixed train-only gradient diagnosis of already-trained round1 weights."""
from pathlib import Path
import itertools
import numpy as np
import torch

from neurokinematics.neural.c104 import ROOT, MLP, load_data, read_json, write_json, sha
from neurokinematics.neural import c106r, c106r_training as training
from neurokinematics.neural.physics import PhysicsLoss


def main():
    c106r.configure()
    out = ROOT / "experiments/C1-06R/round1-analysis"
    config = read_json(out / "gradient-probe-config.json")
    if (out / "gradient-probe.json").exists():
        raise FileExistsError("preserve previous diagnosis")
    audit = read_json(out / "audit.json")
    c106r.guard_hashes(read_json(ROOT / "experiments/C1-06R/training-freeze.json")["files"])
    train, _ = load_data(label_fk=False)
    groups = {}
    for name in config["groups"]:
        family, mode = name.split('/')
        idx = np.flatnonzero((train.family == family) & (train.mode == mode) & train.label_present)
        idx = idx[np.argsort(train.pair_id[idx], kind="stable")][:config["n_per_group"]]
        assert len(idx) == config["n_per_group"]
        groups[name] = train.take(idx)
    results = []
    physics = PhysicsLoss()
    campaign_root = ROOT / "data/generated/C1-06R/round1"
    for seed in config["seeds"]:
        for arm in config["arms"]:
            folder = campaign_root / f"seed-{seed}" / arm
            pointer = read_json(folder / "checkpoint.json")
            for tag in config["checkpoints"]:
                path = folder / ("best.pt" if tag == "best" else pointer["file"])
                assert sha(path) == audit["manifest"][path.relative_to(campaign_root).as_posix()]
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                model = MLP("conditioned").to("cuda")
                model.load_state_dict(payload["model_state_dict"] if tag == "best" else payload["model"])
                model.eval()
                params = tuple(model.parameters())
                for name, rows in groups.items():
                    data = training.tensors(rows, "cuda")
                    z = training.normalized(model(data["x"]), arm)
                    terms, _, _ = physics.components(z, data["y"], data["p"], data["R"])
                    grads = {}
                    for term in config["terms"]:
                        grad = torch.autograd.grad(terms[term].mean(), params, retain_graph=True)
                        grads[term] = torch.cat([g.flatten() for g in grad]).double()
                        assert torch.isfinite(grads[term]).all()
                    norms = {k: float(torch.linalg.vector_norm(g)) for k, g in grads.items()}
                    cos = {f"{a}:{b}": float(torch.dot(grads[a], grads[b]) / (norms[a]*norms[b]))
                           for a, b in itertools.combinations(config["terms"], 2)}
                    assert all(-1.000001 <= v <= 1.000001 for v in cos.values())
                    results.append(dict(seed=seed, arm=arm, checkpoint=tag, group=name,
                                        checkpoint_sha256=sha(path), pair_ids=rows.pair_id.tolist(),
                                        loss={k: float(terms[k].mean().detach()) for k in config["terms"]},
                                        gradient_norm=norms, cosine=cos,
                                        cancellation_ratio=float(torch.linalg.vector_norm(sum(grads.values())) / sum(norms.values()))))
    write_json(out / "gradient-probe.json", dict(status="PASS", config_sha256=sha(out / "gradient-probe-config.json"),
               code_sha256=sha(Path(__file__)), probes=results, trained_updates=0,
               interpretation="Post-hoc train diagnosis; gradient conflict is not proof of causal attribution"))
    print(f"PASS: {len(results)} fixed train gradient probes; no training", flush=True)


if __name__ == "__main__":
    main()
