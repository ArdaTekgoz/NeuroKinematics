"""Preregistered same-data/same-model diagnostic optimizer follow-up."""
import json
import argparse
from pathlib import Path
import time
import torch
from neurokinematics.neural.c104 import MLP, load_data, read_json, write_json, sha, ROOT
from neurokinematics.neural.c106r import configure, subsets, geometric_metrics, PhysicsLoss, CONFIG


def main():
    config = configure()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, default=ROOT / "experiments/C1-06R/overfit-followup-config.json")
    parser.add_argument('--output', type=Path, default=ROOT / "experiments/C1-06R/r0r1/diagnostic-r2")
    args = parser.parse_args()
    protocol = args.protocol
    follow = read_json(protocol)
    source = read_json(ROOT / follow["source_result"])
    weight = ROOT / source["weight"]["path"]
    if sha(weight) != follow["checkpoint_sha256"]:
        raise ValueError("follow-up input hash mismatch")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", dict(protocol=follow, protocol_sha256=sha(protocol), source_sha256=sha(weight), code_sha256=sha(__import__('pathlib').Path(__file__))))
    train, _ = load_data(label_fk=False)
    rows = subsets(train, config)["local64"]
    payload = torch.load(weight, weights_only=True)
    if payload["pair_ids"] != rows.pair_id.tolist() or payload["config_sha256"] != sha(CONFIG):
        raise ValueError("original subset/config drift")
    model = MLP("conditioned").cuda()
    model.load_state_dict(payload["state_dict"])
    x = torch.tensor(rows.conditioned, device="cuda")
    y = torch.tensor(rows.target_normalized, device="cuda")
    options = {k: v for k, v in follow["optimizer"].items() if k != "name"}
    optimizer = torch.optim.LBFGS(model.parameters(), **options)
    calls = 0
    started = time.perf_counter()
    with (output / "closure-losses.jsonl").open("x", encoding="utf-8") as log:
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            raw_loss = ((model(x)-y)**2).sum(-1).mean()
            loss = raw_loss * follow.get('loss_scale', 1.0)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite diagnostic loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite diagnostic gradient")
            calls += 1
            log.write(json.dumps(dict(call=calls, loss=float(raw_loss.detach()), optimizer_loss=float(loss.detach()))) + "\n")
            return loss
        optimizer.step(closure)
    physics = PhysicsLoss()
    with torch.no_grad():
        z = model(x)
        q = physics.raw(z).cpu().numpy()
        loss = float(((z-y)**2).sum(-1).mean())
    result = geometric_metrics(q, rows, details=True)
    result.update(status="PASS" if result["profile_a"] == 64 else "FAIL", closure_calls=calls,
                  optimizer_iterations=optimizer.state[next(iter(model.parameters()))]["n_iter"],
                  q_loss=loss, wall_s=time.perf_counter()-started, protocol_sha256=sha(protocol),
                  source_checkpoint_sha256=sha(weight), prior_profile_a=source["profile_a"],
                  interpretation=follow["interpretation"], long_training="NOT_RUN")
    write_json(output / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "profile_a", "q_loss", "closure_calls", "wall_s")}))
    if result["status"] != "PASS":
        raise ValueError("same-objective refinement failed")


if __name__ == "__main__":
    main()
