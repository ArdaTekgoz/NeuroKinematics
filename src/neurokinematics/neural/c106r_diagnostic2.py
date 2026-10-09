"""Fixed small causal controls after round1; no changes to historical kernels."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .c104 import ROOT, MLP, load_data, read_json, write_json, sha, setup
from .c106r import configure, guard_hashes, geometric_metrics
from .c106r_training import state_hash
from .physics import PhysicsLoss

BASE = ROOT / "experiments/C1-06R/diagnostic2"
CONFIG = BASE / "config.json"


def matched_rows(train, n):
    if train.split != "train" or n % 2:
        raise ValueError("even-sized train-only diagnosis required")
    maps = {mode: {str(train.source_sample_id[i]): i for i in np.flatnonzero(
        (train.family == "main") & (train.mode == mode) & train.label_present)} for mode in ("local", "wide")}
    roots = sorted(set(maps["local"]) & set(maps["wide"]))[:n]
    if len(roots) != n:
        raise ValueError("insufficient matched roots")
    local = train.take(np.array([maps["local"][r] for r in roots]))
    mixed = train.take(np.array([maps["wide" if i % 2 else "local"][r] for i, r in enumerate(roots)]))
    return {"local": local, "mixed": mixed}, local.take(np.arange(0, n, 2))


def build_model(seed, device="cuda"):
    setup(seed)
    model = MLP("conditioned").to(device)
    torch.nn.init.zeros_(model.layers[-1].weight)
    torch.nn.init.zeros_(model.layers[-1].bias)
    return model


def predict(model, x, head):
    if head not in ("absolute", "residual"):
        raise ValueError("unknown head")
    return model(x) + (x[:, -6:] if head == "residual" else .5)


def evaluate(model, head, rows, device="cuda"):
    model.eval()
    physics = PhysicsLoss()
    q = []
    with torch.no_grad():
        for i in range(0, len(rows.pair_id), 1024):
            x = torch.tensor(rows.conditioned[i:i+1024], device=device)
            q.append(physics.raw(predict(model, x, head)).cpu().numpy())
    return geometric_metrics(np.concatenate(q), rows, details=True)


def summarize(metric):
    return {k: v for k, v in metric.items() if k != "rows"}


def checkpoint_contract(config, head, rows):
    # Store str, not TorchVersion; weights_only=True must load without allowlists.
    return dict(torch=str(torch.__version__), cuda=str(torch.version.cuda), seed=config["seed"],
                head=head, config_sha256=sha(CONFIG), code_sha256=sha(Path(__file__)),
                pair_ids=rows.pair_id.tolist(), task="C1-06R-diagnostic2")


def run():
    configure()
    guard_hashes(read_json(ROOT / "experiments/C1-06R/training-freeze.json")["files"])
    if (BASE / "results.json").exists():
        raise FileExistsError("preserve previous diagnosis")
    config = read_json(CONFIG)
    train, validation = load_data(label_fk=True)
    cells, initial = [], None
    write_json(BASE / "registration.json", dict(config_sha256=sha(CONFIG),
        code_sha256=sha(Path(__file__)), status="REGISTERED_BEFORE_TRAINING", final_test="NOT_CREATED"))
    for n in config["sizes"]:
        subsets, shared = matched_rows(train, n)
        for mixture in config["mixtures"]:
            rows = subsets[mixture]
            for head in config["heads"]:
                name = f"n{n}-{mixture}-{head}"
                path = ROOT / "data/generated/C1-06R/diagnostic2" / (name + ".pt")
                if path.exists() or (BASE / (name + ".json")).exists():
                    raise FileExistsError(name)
                tick = time.perf_counter()
                model = build_model(config["seed"])
                digest = state_hash(model)
                initial = digest if initial is None else initial
                assert digest == initial
                x = torch.tensor(rows.conditioned, device="cuda")
                y = torch.tensor(rows.target_normalized, device="cuda")
                optimizer = torch.optim.AdamW(model.parameters(), **{k: v for k, v in config["optimizer"].items() if k != "name"})
                schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"], eta_min=config["schedule"]["eta_min"])
                history = []
                for step in range(config["steps"] + 1):
                    if step:
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        loss = ((predict(model, x, head)-y)**2).sum(-1).mean()
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite loss")
                        loss.backward()
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise ValueError("missing/nonfinite gradient")
                        optimizer.step()
                        schedule.step()
                    if step % config["log_every"] == 0:
                        with torch.no_grad():
                            q_loss = float(((predict(model, x, head)-y)**2).sum(-1).mean())
                        metric = evaluate(model, head, rows)
                        history.append(dict(step=step, q_loss=q_loss, **summarize(metric)))
                final = evaluate(model, head, rows)
                val = evaluate(model, head, validation)
                contract = checkpoint_contract(config, head, rows)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(contract=contract, model_state_dict={k: v.detach().cpu() for k,v in model.state_dict().items()}), path)
                saved = torch.load(path, weights_only=True)
                assert type(saved["contract"]["torch"]) is str and saved["contract"] == contract
                restored = build_model(config["seed"])
                restored.load_state_dict(saved["model_state_dict"])
                assert evaluate(restored, head, rows) == final
                result = dict(name=name, n=n, mixture=mixture, head=head, seed=config["seed"], steps=config["steps"],
                              initial_hash=digest, history=history, train=final, validation=val,
                              shared_local=summarize(evaluate(model, head, shared)),
                              small_gate=("PASS" if final["profile_a"] == n else "FAIL") if n == 64 else "SCALE_DIAGNOSIS",
                              wall_s=time.perf_counter()-tick, contract=contract,
                              checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)), reload="EXACT_MATCH")
                # Full predictions stay outside Git; compact, hash-addressed evidence is tracked.
                raw = path.with_suffix('.json')
                write_json(raw, dict(train=final, validation=val))
                result["raw"] = dict(path=str(raw.relative_to(ROOT)), sha256=sha(raw))
                result["train"], result["validation"] = summarize(final), summarize(val)
                write_json(BASE / (name + ".json"), result)
                cells.append(result)
                print(f"{name}: train A {final['profile_a']}/{n}, validation A {val['profile_a']}/3600, {result['small_gate']}", flush=True)
    write_json(BASE / "results.json", dict(status="COMPLETE_DIAGNOSIS", cells=cells,
        config_sha256=sha(CONFIG), code_sha256=sha(Path(__file__)),
        small_gate="PASS" if all(c["small_gate"] == "PASS" for c in cells if c["n"] == 64) else "FAIL",
        long_training="NOT_RUN", final_test="NOT_CREATED"))


if __name__ == "__main__":
    run()
