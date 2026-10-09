"""C1-06R diagnostics. Only train/validation and frozen FK fixtures are read."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import copy
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .c104 import (ROOT, MLP, Rows, load_data, read_json, write_json, sha,
                   setup, validate_rows, reject_shifted_labels, DATA_ROOT,
                   PAIR_MANIFEST, PAIR_SCHEMA, NORMALIZATION)
from .physics import PhysicsLoss, quaternion_matrix, normalized_head
from neurokinematics.data.factory import read_shard
from neurokinematics.data.pairs import make_base, process_peak_rss_bytes
from neurokinematics.kinematics.model import load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.metrics import rotation_error, quaternion_rotation
from neurokinematics.kinematics.torch_fk import TorchFK

BASE = ROOT / "experiments/C1-06R"
CONFIG = BASE / "r0r1-config.json"


def configure():
    config = read_json(CONFIG)
    setup(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("run via c106r_command.py to enforce deterministic environment")
    return config


def guard_hashes(mapping):
    records = []
    for path, expected in mapping.items():
        source = ROOT / path
        actual = sha(source)
        if actual != expected:
            raise ValueError(f"input SHA mismatch: {path}")
        records.append(dict(path=path, bytes=source.stat().st_size, sha256=actual))
    return records


def preflight(output):
    config = configure()
    files = guard_hashes(read_json(ROOT / "experiments/F0-06/handoff-inputs.json"))
    # Bind accepted C1-02 source metadata, code and CPU acceptance fixtures.
    extra = [CONFIG, BASE / "requirements-win-cu128.lock", PAIR_MANIFEST, PAIR_SCHEMA,
             NORMALIZATION, ROOT / "experiments/C1-03/config.json",
             ROOT / config["fk_samples"], Path(__file__),
             ROOT / "src/neurokinematics/neural/c104.py",
             ROOT / "src/neurokinematics/neural/physics.py",
             ROOT / "src/neurokinematics/neural/training_fk.py",
             ROOT / "src/neurokinematics/kinematics/torch_fk.py"]
    files += [dict(path=str(p.relative_to(ROOT)).replace("\\", "/"), bytes=p.stat().st_size, sha256=sha(p)) for p in extra]
    manifest = read_json(PAIR_MANIFEST)
    shards = {str((DATA_ROOT / s["path"]).relative_to(ROOT)).replace("\\", "/"): s["file_sha256"]
              for s in manifest["shards"] if "-train-" in s["path"] or "-validation-" in s["path"]}
    files += guard_hashes(shards)
    record = dict(status="PASS", files=files, train_validation_shards=len(shards),
                  config_sha256=sha(CONFIG), head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                  free_disk_bytes=shutil.disk_usage(ROOT).free, final_test="NOT_CREATED", old_final_raw="NOT_READ")
    write_json(output / "preflight.json", record)
    return record


def geometric_metrics(q, rows, *, details=False):
    """Full denominator; invalid geometry encoded as null plus invalid counts."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (len(rows.pair_id), 6):
        raise ValueError("q/row shape mismatch")
    robot = load_robot()
    bounds = np.asarray(robot.limits)
    fk = PinocchioFK(robot)
    finite = np.isfinite(q).all(axis=1)
    valid = finite & (q >= bounds[:, 0]).all(axis=1) & (q <= bounds[:, 1]).all(axis=1)
    pe = np.full(len(q), np.inf)
    re = np.full(len(q), np.inf)
    for i in np.flatnonzero(valid):
        t = fk.reference_forward_kinematics(q[i])
        pe[i] = np.linalg.norm(t[:3, 3] - rows.position[i])
        re[i] = math.degrees(rotation_error(t[:3, :3], quaternion_rotation(rows.quaternion[i])))
    a, b = (pe <= .002) & (re <= 1.), (pe <= .001) & (re <= .5)
    def quantiles(values):
        ordered = np.sort(values)
        return {name: float(ordered[math.ceil(p * len(values)) - 1]) if np.isfinite(ordered[math.ceil(p * len(values)) - 1]) else None
                for name, p in [("median", .5), ("p95", .95), ("p99", .99), ("max", 1)]}
    result = dict(n=len(q), profile_a=int(a.sum()), profile_b=int(b.sum()),
                  nonfinite=int((~finite).sum()), out_of_limits=int((finite & ~valid).sum()),
                  valid=int(valid.sum()), position_m=quantiles(pe), orientation_deg=quantiles(re),
                  quantile_policy="full denominator nearest-rank; invalid +inf encoded null",
                  by_family_mode={})
    for group in sorted(set(zip(rows.family.tolist(), rows.mode.tolist()))):
        mask = (rows.family == group[0]) & (rows.mode == group[1])
        result["by_family_mode"]["/".join(group)] = dict(n=int(mask.sum()), profile_a=int(a[mask].sum()), profile_b=int(b[mask].sum()))
    if details:
        result["rows"] = [dict(pair_id=str(rows.pair_id[i]), q_rad=q[i].tolist() if finite[i] else None,
                               valid=bool(valid[i]), profile_a=bool(a[i]), profile_b=bool(b[i]),
                               position_m=float(pe[i]) if valid[i] else None,
                               orientation_deg=float(re[i]) if valid[i] else None) for i in range(len(q))]
    return result


def subsets(train, config):
    if train.split != "train":
        raise ValueError("overfit must use train only")
    def first(family, mode, n):
        indices = np.flatnonzero((train.family == family) & (train.mode == mode) & train.label_present)
        indices = indices[np.argsort(train.pair_id[indices], kind="stable")][:n]
        if len(indices) != n:
            raise ValueError("insufficient overfit stratum")
        return indices
    local = first("main", "local", 64)
    mixed = np.concatenate([first(*group.split("/"), n) for group, n in config["overfit"]["mixed_counts"].items()])
    return {"local64": train.take(local), "mixed64": train.take(mixed)}


def runtime(output):
    config = configure()
    if torch.__version__ != config["runtime"]["torch"] or not torch.cuda.is_available():
        raise ValueError("exact CUDA runtime unavailable")
    versions = {}
    for name in ("torch", "numpy", "setuptools", "typing_extensions", "sympy", "networkx", "fsspec", "mpmath", "filelock", "Jinja2", "MarkupSafe"):
        dist = metadata.distribution(name)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        versions[name] = dict(version=dist.version, location=str(dist.locate_file("")), direct_url=direct)
        if not Path(dist.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError(f"package outside isolated overlay: {name}")
    fixture = [json.loads(line) for line in (ROOT / config["fk_samples"]).read_text().splitlines()]
    robot, fk, physics = load_robot(), TorchFK.from_frozen(), PhysicsLoss()
    pin, independent = PinocchioFK(robot), IndependentFK(robot)
    kw = dict(robot_id=fk.robot_id, joint_names=fk.joint_names)
    evidence = []
    for dtype, key in [(torch.float64, "q64"), (torch.float32, "q32")]:
        values = np.asarray([r[key] for r in fixture])
        refs = np.stack([pin.reference_forward_kinematics(q) for q in values])
        independent_refs = np.stack([independent.forward_kinematics(q) for q in values])
        if np.max(np.abs(refs - independent_refs)) > 1e-9:
            raise ValueError("independent reference disagreement")
        for device in ("cpu", "cuda"):
            q = torch.tensor(values, dtype=dtype, device=device)
            result = fk(q, **kw).detach().cpu().numpy().astype(np.float64)
            extended = physics.fk(q, **kw).detach().cpu().numpy().astype(np.float64)
            pe = np.linalg.norm(result[:, :3, 3] - refs[:, :3, 3], axis=1)
            re = np.linalg.norm(result[:, :3, :3] - refs[:, :3, :3], axis=(1, 2))
            tolerance = config["fk_tolerances"][str(dtype).split(".")[-1]]
            passed = bool((pe <= tolerance).all() and (re <= tolerance).all() and np.max(np.abs(result - extended)) <= tolerance)
            evidence.append(dict(dtype=str(dtype), device=device, n=len(q), position_max=float(pe.max()), rotation_max=float(re.max()), threshold=tolerance, passed=passed))
    gradient = []
    grad_values = [r["q64"] for r in fixture if r["group"] == "gradient"]
    if len(grad_values) < config["gradient"]["samples"]:
        # Fixture uses group 'grad' in some accepted revisions; exact IDs identify these rows.
        grad_values = [r["q64"] for r in fixture if r["id"].startswith("grad-")]
    if len(grad_values) != config["gradient"]["samples"]:
        raise ValueError("gradient fixture count")
    eps = config["gradient"]["epsilon"]
    def vector(t):
        return np.concatenate((t[:3, 3], t[:3, :3].reshape(-1)))
    for device in ("cpu", "cuda"):
        for i, values in enumerate(grad_values):
            q = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)
            def outputs(x):
                t = fk(x, **kw)
                return torch.cat((t[:3, 3], t[:3, :3].reshape(-1)))
            jac = torch.autograd.functional.jacobian(outputs, q).detach().cpu().numpy()
            fd = np.empty_like(jac)
            for j in range(6):
                plus, minus = np.array(values), np.array(values)
                plus[j] += eps; minus[j] -= eps
                fd[:, j] = (vector(pin.reference_forward_kinematics(plus)) - vector(pin.reference_forward_kinematics(minus))) / (2 * eps)
            allowed = config["gradient"]["atol"] + config["gradient"]["rtol"] * np.abs(fd)
            gradient.append(dict(device=device, index=i, max_abs=float(np.max(np.abs(jac - fd))), passed=bool((np.abs(jac - fd) <= allowed).all())))
    cpu_model = MLP("conditioned")
    gpu_model = copy.deepcopy(cpu_model).cuda()
    x = torch.randn(64, 13)
    y_cpu, y_gpu = cpu_model(x), gpu_model(x.cuda())
    cpu_loss, gpu_loss = y_cpu.square().mean(), y_gpu.square().mean()
    cpu_loss.backward(); gpu_loss.backward()
    parity = float((y_cpu - y_gpu.cpu()).abs().max().detach())
    grad_parity = max(float((p.grad - g.grad.cpu()).abs().max()) for p, g in zip(cpu_model.parameters(), gpu_model.parameters()))
    probe = physics_probe()
    passed = all(x["passed"] for x in evidence + gradient) and parity <= config["inference_normalized_atol"] and grad_parity <= config["inference_normalized_atol"] and probe["status"] == "PASS"
    result = dict(status="PASS" if passed else "FAIL", torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), arch_list=torch.cuda.get_arch_list(),
                  python=sys.version, platform=platform.platform(), executable=sys.executable, packages=versions,
                  robot_hashes=robot.hashes, fk=evidence, gradients=gradient, mlp_forward_max=parity,
                  mlp_gradient_max=grad_parity, physics_probe=probe, peak_cuda_bytes=torch.cuda.max_memory_allocated(), config_sha256=sha(CONFIG))
    write_json(output / "runtime.json", result)
    if not passed:
        raise ValueError("runtime numerical gate failed")
    return result


def physics_probe():
    """Independent normalized-coordinate finite differences for training losses."""
    robot = load_robot()
    lower, upper = np.asarray(robot.limits).T
    span = upper - lower
    pin = PinocchioFK(robot)
    z_np = np.array([.35, .42, .51, .61, .47, .58])
    target_np = z_np + np.array([.01, -.02, .03, -.01, .02, -.03])
    target_t = pin.reference_forward_kinematics(lower + target_np * span)
    physics = PhysicsLoss()
    z = torch.tensor(z_np[None], dtype=torch.float64, device="cuda", requires_grad=True)
    target = torch.tensor(target_np[None], dtype=torch.float64, device="cuda")
    p = torch.tensor(target_t[None, :3, 3], dtype=torch.float64, device="cuda")
    r = torch.tensor(target_t[None, :3, :3], dtype=torch.float64, device="cuda")
    terms, _, _ = physics.components(z, target, p, r)
    def reference(values):
        t = pin.reference_forward_kinematics(lower + values * span)
        return dict(q=float(((values-target_np)**2).sum()),
                    p=float((((t[:3, 3]-target_t[:3, 3])/.9015)**2).sum()),
                    R=float(((t[:3, :3]-target_t[:3, :3])**2).sum()/8))
    result = {}
    passed = True
    for key in ("q", "p", "R"):
        analytic = torch.autograd.grad(terms[key].sum(), z, retain_graph=True)[0].detach().cpu().numpy()[0]
        fd = np.empty(6)
        for j in range(6):
            plus, minus = z_np.copy(), z_np.copy()
            plus[j] += 1e-6; minus[j] -= 1e-6
            fd[j] = (reference(plus)[key] - reference(minus)[key]) / 2e-6
        good = bool(np.isfinite(analytic).all() and np.linalg.norm(analytic) > 0 and (np.abs(analytic-fd) <= 1e-5+1e-3*np.abs(fd)).all())
        result[key] = dict(value=float(terms[key].detach()), gradient_norm=float(np.linalg.norm(analytic)), fd_max_abs=float(np.abs(analytic-fd).max()), passed=good)
        passed &= good
    logits = torch.atanh(2*z-1)
    reconstructed = normalized_head(logits, "FK_TANH")
    roundtrip = float((reconstructed-z).abs().max().detach())
    tanh_grad = torch.autograd.grad(physics.raw(normalized_head(logits, "FK_TANH")).sum(), logits)[0]
    expected = torch.tensor(span, device="cuda")/2 * (1-torch.tanh(logits.detach())**2)
    derivative_error = float((tanh_grad-expected).abs().max())
    # Actual MLP -> physical FK -> loss -> parameter update, without label loss.
    model = MLP("conditioned").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    features = torch.randn(4, 13, device="cuda")
    before = [p.detach().clone() for p in model.parameters()]
    pred = model(features)
    parts, _, _ = physics.components(pred, torch.zeros_like(pred), p.float().expand(4, -1), r.float().expand(4, -1, -1))
    task_loss = (parts["p"] + parts["R"]).mean()
    task_loss.backward()
    finite_gradients = all(param.grad is not None and bool(torch.isfinite(param.grad).all()) for param in model.parameters())
    optimizer.step()
    update_norm = math.sqrt(sum(float(((param.detach()-old).double()**2).sum()) for param, old in zip(model.parameters(), before)))
    passed &= roundtrip <= 1e-12 and derivative_error <= 1e-12 and finite_gradients and update_norm > 0
    return dict(status="PASS" if passed else "FAIL", components=result, tanh_roundtrip_max=roundtrip,
                tanh_derivative_max=derivative_error, fk_only_parameter_update_norm=update_norm,
                finite_parameter_gradients=finite_gradients)


def audit_data(output):
    config = configure()
    train, val = load_data(label_fk=True)
    norm = read_json(NORMALIZATION)
    mean, std = train.position.mean(0), train.position.std(0)
    if not np.allclose(mean, norm["position_mean_m"], atol=1e-12, rtol=0) or not np.allclose(std, norm["position_std_m"], atol=1e-12, rtol=0):
        raise ValueError("train-only normalization mismatch")
    raw_records, shard_hashes = [], []
    fields = [x["name"] for x in read_json(PAIR_SCHEMA)["fields"]]
    pair_config = read_json(ROOT / "experiments/C1-02/config.json")
    bounds = np.asarray(load_robot().limits)
    pin = PinocchioFK(load_robot())
    for shard in read_json(PAIR_MANIFEST)["shards"]:
        if not any(f"-{s}-" in shard["path"] for s in config["splits"]):
            continue
        path = DATA_ROOT / shard["path"]
        arrays = read_shard(path, fields)
        shard_hashes.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        for i in range(shard["record_count"]):
            def text(name): return arrays[name][i].decode()
            root = dict(sample_id=text("source_sample_id"), group_id=text("group_id"), family=text("source_family"), split=text("split"),
                        q=arrays["root_q_target"][i], position=arrays["position_m"][i], quaternion=arrays["quaternion_wxyz"][i])
            base = make_base(root, text("pair_mode"), pair_config, bounds)
            if not np.array_equal(base["q_current"], arrays["q_current"][i]) or base["derivation_seed"] != int(arrays["derivation_seed"][i]) or base["derivation_attempts"] != int(arrays["derivation_attempts"][i]):
                raise ValueError("q_current provenance mismatch")
            t = pin.reference_forward_kinematics(root["q"])
            if np.linalg.norm(t[:3, 3] - root["position"]) > 1e-9 or np.linalg.norm(t[:3, :3] - quaternion_rotation(root["quaternion"])) > 1e-9:
                raise ValueError("root target pose mismatch")
            raw_records.append((root["split"], root["family"], text("pair_mode"), text("teacher_status"), text("teacher_failure_class")))
    local = subsets(train, config)["local64"]
    negatives = dict(shifted_labels=reject_shifted_labels(local))
    for name, altered in [("feature_order", replace(local, conditioned=local.conditioned[:, ::-1].copy())),
                           ("target_scaling", replace(local, target_normalized=local.target_normalized + 1)),
                           ("nonfinite_current", replace(local, q_current=np.full_like(local.q_current, np.nan)))]:
        try:
            validate_rows(altered)
        except ValueError as exc:
            negatives[name] = str(exc)
        else:
            raise ValueError("negative control survived: " + name)
    inventory = {"/".join(key): n for key, n in sorted(Counter(raw_records).items())}
    metrics = {}
    for rows in (train, val):
        labeled = rows.take(np.flatnonzero(rows.label_present))
        counts = dict(total=len(rows.pair_id), labeled=int(rows.label_present.sum()), missing=int((~rows.label_present).sum()))
        counts["teacher_geometry"] = geometric_metrics(labeled.q_target, labeled)
        counts["input_current_geometry"] = geometric_metrics(rows.q_current, rows)
        counts["duplicate_conditioned_inputs"] = len(rows.pair_id) - len(np.unique(rows.conditioned, axis=0))
        metrics[rows.split] = counts
    witnesses = {name: r.pair_id.tolist() for name, r in subsets(train, config).items()}
    result = dict(status="PASS", splits=metrics, inventory=inventory, shards=shard_hashes,
                  provenance_rows=len(raw_records), root_overlap=0, group_overlap=0,
                  normalization_max_error=float(max(np.max(np.abs(mean - norm["position_mean_m"])), np.max(np.abs(std - norm["position_std_m"])))),
                  negatives=negatives, overfit_pair_ids=witnesses, test_shards="NOT_READ", config_sha256=sha(CONFIG))
    write_json(output / "data-audit.json", result)
    return result


def overfit(output):
    config = configure()
    if read_json(output / "runtime.json")["status"] != "PASS" or read_json(output / "data-audit.json")["status"] != "PASS":
        raise ValueError("runtime/data gates required")
    guard_hashes({x["path"]: x["sha256"] for x in read_json(output / "preflight.json")["files"]})
    train, _ = load_data(label_fk=False)
    runs = {}
    spec = config["overfit"]
    for name, rows in subsets(train, config).items():
        setup(config["seed"])
        model = MLP("conditioned").cuda()
        x = torch.tensor(rows.conditioned, device="cuda")
        target = torch.tensor(rows.target_normalized, device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec["steps"], eta_min=.00001)
        physics = PhysicsLoss()
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with (output / (name + "-epochs.jsonl")).open("x", encoding="utf-8") as log:
            for step in range(spec["steps"] + 1):
                if step:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss = ((model(x) - target)**2).sum(-1).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite overfit loss")
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError("overfit gradient missing/nonfinite")
                    optimizer.step(); schedule.step()
                if step % spec["log_every"] == 0:
                    model.eval()
                    with torch.no_grad():
                        z = model(x)
                        physical = physics.raw(z).cpu().numpy()
                        measured_loss = float(((z - target)**2).sum(-1).mean())
                    metric = geometric_metrics(physical, rows)
                    log.write(json.dumps(dict(step=step, q_loss=measured_loss, lr=optimizer.param_groups[0]["lr"], **metric), allow_nan=False) + "\n")
                    log.flush()
        torch.cuda.synchronize()
        final = geometric_metrics(physical, rows, details=True)
        weight = ROOT / "data/generated/C1-06R/diagnostics" / output.name / (name + "-last.pt")
        weight.parent.mkdir(parents=True, exist_ok=True)
        if weight.exists():
            raise FileExistsError(weight)
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(dict(state_dict=cpu_state, seed=config["seed"], step=spec["steps"], pair_ids=rows.pair_id.tolist(), config_sha256=sha(CONFIG)), weight)
        restored = MLP("conditioned")
        restored.load_state_dict(torch.load(weight, weights_only=True)["state_dict"])
        with torch.no_grad():
            error = float((restored(torch.tensor(rows.conditioned)) - z.cpu()).abs().max())
        final.update(status="PASS" if final["profile_a"] == spec["required_profile_a"] and error <= config["inference_normalized_atol"] else "FAIL",
                     steps=spec["steps"], wall_s=time.perf_counter()-start, q_loss=measured_loss,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated(), peak_process_ram_bytes=process_peak_rss_bytes(),
                     cpu_reload_normalized_max=error, weight=dict(path=str(weight.relative_to(ROOT)), bytes=weight.stat().st_size, sha256=sha(weight)),
                     generalization="NOT_CLAIMED")
        write_json(output / (name + "-result.json"), final)
        runs[name] = final
    result = dict(status="PASS" if all(r["status"] == "PASS" for r in runs.values()) else "FAIL", runs=runs, config_sha256=sha(CONFIG), long_training="NOT_RUN")
    write_json(output / "overfit.json", result)
    if result["status"] != "PASS":
        raise ValueError("overfit gate failed; long training blocked")
    return result
