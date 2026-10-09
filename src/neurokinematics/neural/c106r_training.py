"""Paired validation-only C1-06R training with atomic, reproducible resume."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .c104 import ROOT, MLP, Rows, load_data, read_json, write_json, sha, setup
from .c106r import configure, geometric_metrics, guard_hashes
from .physics import PhysicsLoss, quaternion_matrix

PROTOCOL = ROOT / "experiments/C1-06R/training-round1.json"


def normalized(logits, arm):
    if arm not in ("Q", "FK", "Q_TANH", "FK_TANH"):
        raise ValueError("unknown round1 arm")
    return (torch.tanh(logits)+1)/2 if arm.endswith("TANH") else logits


def atomic_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, payload)
    temp.replace(path)


def save_state(directory, payload):
    """Publish an inactive slot before its hash pointer; keep previous valid slot."""
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / "checkpoint.json"
    slot = 1 - read_json(pointer)["slot"] if pointer.exists() else 0
    target = directory / f"last-{slot}.pt"
    temp = target.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    digest, size = sha(temp), temp.stat().st_size
    temp.replace(target)
    atomic_json(pointer, dict(slot=slot, file=target.name, epoch=payload["epoch"], sha256=digest, bytes=size))


def load_state(directory, expected_contract):
    pointer = read_json(directory / "checkpoint.json")
    if pointer["slot"] not in (0, 1) or pointer["file"] != f"last-{pointer['slot']}.pt":
        raise ValueError("invalid checkpoint slot")
    target = directory / pointer["file"]
    if sha(target) != pointer["sha256"] or target.stat().st_size != pointer["bytes"]:
        raise ValueError("checkpoint integrity failure")
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if payload["contract"] != expected_contract or payload["epoch"] != pointer["epoch"]:
        raise ValueError("resume contract/epoch mismatch")
    return payload


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_rank(metrics):
    rows = metrics["rows"]
    if len(rows) != metrics["n"] or not rows:
        raise ValueError("full validation denominator required")
    error = math.fsum((r["position_m"]/.002)**2 + r["orientation_deg"]**2 if r["valid"] else 1e12 for r in rows) / len(rows)
    return [-metrics["profile_a"], metrics["nonfinite"] + metrics["out_of_limits"], error]


def tensors(rows, device):
    return dict(x=torch.tensor(rows.conditioned, device=device),
                y=torch.tensor(rows.target_normalized, device=device),
                p=torch.tensor(rows.position, dtype=torch.float32, device=device),
                R=quaternion_matrix(torch.tensor(rows.quaternion, dtype=torch.float32, device=device)))


def loss_parts(model, arm, data, indices, physics, spec):
    z = normalized(model(data["x"][indices]), arm)
    if arm.startswith("Q"):
        parts = dict(q=((z-data["y"][indices])**2).sum(-1))
        return parts["q"].mean(), parts
    parts, _, _ = physics.components(z, data["y"][indices], data["p"][indices], data["R"][indices])
    total = parts["q"] + spec["loss"]["lambda_p"]*parts["p"] + spec["loss"]["lambda_R"]*parts["R"]
    return total.mean(), parts


def evaluate(model, arm, rows, device):
    physics = PhysicsLoss()
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(rows.pair_id), 1024):
            z = normalized(model(torch.tensor(rows.conditioned[start:start+1024], device=device)), arm)
            values.append(physics.raw(z).cpu().numpy())
    return geometric_metrics(np.concatenate(values), rows, details=True)


def train_run(train, validation, spec, arm, seed, directory, contract, *, device="cuda", resume=False, stop_after=None):
    """stop_after exists for resume verification; it does not alter the frozen budget."""
    if train.split != "train" or validation.split != "validation":
        raise ValueError("train/validation only")
    if arm not in spec["arms"] or seed not in spec["seed_list"]:
        raise ValueError("unregistered arm/seed")
    labeled = train.take(np.flatnonzero(train.label_present))
    setup(seed)
    model = MLP("conditioned").to(device)
    initial_hash = state_hash(model)
    optspec = spec["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optspec["lr"], betas=tuple(optspec["betas"]), eps=optspec["eps"], weight_decay=optspec["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec["epochs"], eta_min=spec["schedule"]["eta_min"])
    epoch, best, best_state, history = 0, None, None, []
    if directory.exists() and not resume:
        raise FileExistsError("existing run requires --resume")
    if resume:
        payload = load_state(directory, contract)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        schedule.load_state_dict(payload["scheduler"])
        torch.set_rng_state(payload["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        epoch, best, best_state, history = payload["epoch"], payload["best"], payload["best_state"], payload["history"]
        if payload["initial_hash"] != initial_hash:
            raise ValueError("initial tensor state changed")
    data = tensors(labeled, device)
    physics = PhysicsLoss()
    start = time.perf_counter()
    last_epoch = min(spec["epochs"], stop_after) if stop_after is not None else spec["epochs"]
    for epoch in range(epoch+1, last_epoch+1):
        tick = time.perf_counter()
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch]))).permutation(len(labeled.pair_id))
        model.train()
        sums = {}
        updates = 0
        for offset in range(0, len(order), spec["batch_size"]):
            idx = torch.tensor(order[offset:offset+spec["batch_size"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = loss_parts(model, arm, data, idx, physics, spec)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if not bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()) or any(p.grad is None for p in model.parameters()):
                raise ValueError("missing/nonfinite parameter gradient")
            optimizer.step()
            for key, values in parts.items():
                sums[key] = sums.get(key, 0) + values.detach().double().sum()
            updates += 1
        schedule.step()
        record = dict(epoch=epoch, updates=updates, samples=len(order), permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                      loss={k: float(v/len(order)) for k,v in sums.items()}, lr=optimizer.param_groups[0]["lr"], wall_s=time.perf_counter()-tick)
        if epoch % spec["validation_every_epochs"] == 0 or epoch == spec["epochs"]:
            measured = evaluate(model, arm, validation, device)
            rank = checkpoint_rank(measured)
            measured.pop("rows")
            record["validation"] = measured
            record["selection_rank"] = rank
            if best is None or rank < best["rank"]:
                best = dict(epoch=epoch, rank=rank, validation=measured)
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(record)
        payload = dict(contract=contract, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                       python_rng=random.getstate(), initial_hash=initial_hash, best=best, best_state=best_state, history=history)
        save_state(directory, payload)
        # checkpoint history is authoritative; this human-readable mirror is rebuilt on resume.
        temp = directory / "epochs.jsonl.tmp"
        temp.write_text("".join(json.dumps(x, allow_nan=False)+"\n" for x in history), encoding="utf-8")
        temp.replace(directory / "epochs.jsonl")
        if epoch == 1 or epoch % spec["validation_every_epochs"] == 0:
            print(json.dumps(dict(arm=arm, seed=seed, epoch=epoch, epochs=spec["epochs"], loss=record["loss"], best=best)), flush=True)
    complete = last_epoch == spec["epochs"]
    result = dict(status="COMPLETE" if complete else "INTERRUPTED_FOR_RESUME_TEST", arm=arm, seed=seed, epoch=last_epoch,
                  initial_hash=initial_hash, final_state_hash=state_hash(model), best=best, history=history,
                  this_invocation_wall_s=time.perf_counter()-start, contract=contract)
    if complete:
        if best_state is None:
            raise ValueError("completed run has no eligible checkpoint")
        best_path = directory / "best.pt"
        temp = directory / "best.pt.tmp"
        torch.save(dict(contract=contract, model_state_dict=best_state, best=best), temp)
        temp.replace(best_path)
        model.load_state_dict(best_state)
        full = evaluate(model, arm, validation, device)
        atomic_json(directory / "best-validation.json", full)
        result["best_checkpoint"] = dict(path=str(best_path), bytes=best_path.stat().st_size, sha256=sha(best_path))
        atomic_json(directory / "complete.json", result)
    return result


def production_contract(spec, arm, seed):
    paths = [PROTOCOL, ROOT/"pixi.lock", ROOT/"experiments/C1-06R/requirements-win-cu128.lock",
             ROOT/"experiments/C1-02/dataset-manifest.json", ROOT/"experiments/C1-02/normalization.json",
             ROOT/"src/neurokinematics/neural/c104.py", ROOT/"src/neurokinematics/neural/physics.py",
             ROOT/"src/neurokinematics/neural/training_fk.py", ROOT/"src/neurokinematics/neural/c106r.py", Path(__file__)]
    return dict(task="C1-06R", round="round1", arm=arm, seed=seed, torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                hashes={str(p.relative_to(ROOT)).replace("\\", "/"):sha(p) for p in paths})


def production(arm, seed, *, resume):
    configure()
    frozen = read_json(ROOT/"experiments/C1-06R/training-freeze.json")
    guard_hashes(frozen["files"])
    if frozen["status"] != "READY_FOR_USER_TRAINING":
        raise ValueError("training handoff gate not ready")
    if torch.__version__ != "2.10.0+cu128" or not torch.cuda.is_available():
        raise ValueError("exact CUDA environment required")
    spec = read_json(PROTOCOL)
    if arm not in spec["arms"] or seed not in spec["seed_list"]:
        raise ValueError("unregistered request")
    train, validation = load_data(label_fk=True)
    contract = production_contract(spec, arm, seed)
    directory = ROOT/"data/generated/C1-06R/round1"/f"seed-{seed}"/arm
    if resume and (directory/"complete.json").exists():
        result = read_json(directory/"complete.json")
        if result["contract"] != contract or sha(Path(result["best_checkpoint"]["path"])) != result["best_checkpoint"]["sha256"]:
            raise ValueError("completed-run identity mismatch")
        return result
    return train_run(train, validation, spec, arm, seed, directory, contract, resume=resume and directory.exists())
