"""Read-only validation campaign audit; never trains or reads final-test data."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from neurokinematics.neural.c104 import ROOT, MLP, load_data, read_json, write_json, sha
from neurokinematics.neural import c106r, c106r_training as training


def verify_manifest(root):
    entries = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or name in entries:
            raise ValueError("unsafe or duplicate manifest entry")
        if sha(path) != digest:
            raise ValueError("manifest SHA mismatch: " + name)
        entries[name] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    if actual != set(entries):
        raise ValueError("manifest inventory mismatch")
    return entries


def verify_history(history, spec):
    if [h["epoch"] for h in history] != list(range(1, spec["epochs"] + 1)):
        raise ValueError("missing/duplicate/reordered epoch")
    for h in history:
        if h["updates"] != spec["budgets"]["updates_per_epoch"] or h["samples"] != 15204:
            raise ValueError("training budget mismatch")
        if not all(math.isfinite(x) for x in h["loss"].values()):
            raise ValueError("nonfinite loss")
        expected = h["epoch"] % spec["validation_every_epochs"] == 0 or h["epoch"] == spec["epochs"]
        if ("validation" in h) != expected:
            raise ValueError("validation cadence mismatch")
        if expected and h["validation"]["n"] != 3600:
            raise ValueError("validation denominator mismatch")
    eligible = [h for h in history if "validation" in h]
    return min(eligible, key=lambda h: (h["selection_rank"], h["epoch"]))


def summary(metrics, rows):
    records = metrics["rows"]
    result = {k: v for k, v in metrics.items() if k != "rows"}
    groups = {"all": np.ones(len(records), dtype=bool), "labeled": rows.label_present,
              "unlabeled": ~rows.label_present}
    groups.update({f"{f}/{m}": (rows.family == f) & (rows.mode == m)
                   for f, m in sorted(set(zip(rows.family.tolist(), rows.mode.tolist())))})
    groups.update({m: rows.mode == m for m in ("local", "wide")})
    result["groups"] = {}
    for name, mask in groups.items():
        subset = [r for r, keep in zip(records, mask) if keep]
        if not subset:
            continue
        out = dict(n=len(subset), profile_a=sum(r["profile_a"] for r in subset),
                   profile_b=sum(r["profile_b"] for r in subset), invalid=sum(not r["valid"] for r in subset))
        for key in ("position_m", "orientation_deg"):
            values = sorted(r[key] if r["valid"] else math.inf for r in subset)
            out[key] = {label: values[math.ceil(p*len(values))-1] if math.isfinite(values[math.ceil(p*len(values))-1]) else None
                        for label, p in (("median", .5), ("p95", .95), ("p99", .99))}
        label_indices = np.flatnonzero(mask & rows.label_present)
        if len(label_indices):
            q = np.asarray([records[i]["q_rad"] for i in label_indices])
            out["teacher_q_rmse_rad"] = float(np.sqrt(np.mean((q-rows.q_target[label_indices])**2)))
        result["groups"][name] = out
    return result


def main():
    from check_c106r_training import check
    check()
    campaign_root = ROOT / "data/generated/C1-06R/round1"
    out = ROOT / "experiments/C1-06R/round1-analysis"
    if (out / "audit.json").exists():
        raise FileExistsError("preserve previous analysis; use a new revision")
    started = time.perf_counter()
    manifest = verify_manifest(campaign_root)
    campaign = read_json(campaign_root / "campaign-complete.json")
    spec = read_json(training.PROTOCOL)
    expected = {(s, a) for s in spec["seed_list"] for a in spec["arms"]}
    actual = [(r["seed"], r["arm"]) for r in campaign["runs"]]
    assert len(actual) == len(expected) and set(actual) == expected
    assert campaign["status"] == "COMPLETE_VALIDATION_ONLY"
    assert campaign["protocol_sha256"] == sha(training.PROTOCOL)
    train, val = load_data(label_fk=True)
    records, paired = [], {}
    for run in campaign["runs"]:
        seed, arm = run["seed"], run["arm"]
        folder = campaign_root / f"seed-{seed}" / arm
        complete = read_json(folder / "complete.json")
        contract = training.production_contract(spec, arm, seed)
        assert complete["contract"] == contract
        assert complete["status"] == run["status"] == "COMPLETE"
        assert complete["epoch"] == spec["epochs"]
        assert complete["best"] == run["best"]
        assert complete["best_checkpoint"] == run["best_checkpoint"]
        assert sha(folder / "best.pt") == run["best_checkpoint"]["sha256"]
        assert (folder / "best.pt").stat().st_size == run["best_checkpoint"]["bytes"]
        # Historical production metadata stored TorchVersion, a str subclass.
        # Keep weights_only=True and allow only this installed Torch class.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            last = training.load_state(folder, contract)
        assert last["epoch"] == spec["epochs"] and last["history"] == complete["history"]
        import json
        assert [json.loads(line) for line in (folder / "epochs.jsonl").read_text().splitlines()] == last["history"]
        best_record = verify_history(last["history"], spec)
        assert best_record["epoch"] == complete["best"]["epoch"]
        assert best_record["selection_rank"] == complete["best"]["rank"]
        assert best_record["validation"] == complete["best"]["validation"]
        hashes = []
        for h in last["history"]:
            order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, h["epoch"]]))).permutation(15204)
            digest = hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
            assert digest == h["permutation_sha256"]
            hashes.append(digest)
        training.setup(seed)
        model = MLP("conditioned").to("cuda")
        assert training.state_hash(model) == complete["initial_hash"] == last["initial_hash"]
        identity = (complete["initial_hash"], hashes)
        if seed in paired:
            assert paired[seed] == identity
        paired[seed] = identity
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            best = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        assert best["contract"] == contract and best["best"] == complete["best"]
        assert all(torch.equal(v, last["best_state"][k]) for k, v in best["model_state_dict"].items())
        history = last["history"]
        result = dict(seed=seed, arm=arm, best_epoch=best_record["epoch"], epochs=len(history),
                      updates=sum(h["updates"] for h in history), wall_s=complete["this_invocation_wall_s"],
                      epoch_training_s=sum(h["wall_s"] for h in history),
                      max_validation_a=max(h.get("validation", {}).get("profile_a", 0) for h in history),
                      max_validation_b=max(h.get("validation", {}).get("profile_b", 0) for h in history),
                      first_loss=history[0]["loss"], last_loss=history[-1]["loss"], checks="PASS")
        for tag, state in (("best", best["model_state_dict"]), ("last", last["model"])):
            model.load_state_dict(state)
            if tag == "last":
                assert training.state_hash(model) == complete["final_state_hash"]
            measured = training.evaluate(model, arm, val, "cuda")
            if tag == "best":
                assert measured == read_json(folder / "best-validation.json")
            stored = complete["best"]["validation"] if tag == "best" else history[-1]["validation"]
            assert {k: v for k, v in measured.items() if k != "rows"} == stored
            assert training.checkpoint_rank(measured) == (complete["best"]["rank"] if tag == "best" else history[-1]["selection_rank"])
            result[tag] = dict(validation=summary(measured, val),
                               train=summary(training.evaluate(model, arm, train, "cuda"), train))
        records.append(result)
        write_json(out / f"seed-{seed}-{arm}.json", result)
        print(f"PASS {seed}/{arm}: best/last validation reproduced; train A {result['last']['train']['profile_a']}/16800", flush=True)
    # A diagnostic input baseline, not a solver or a new trained candidate.
    baseline = {r.split: summary(c106r.geometric_metrics(r.q_current, r, details=True), r) for r in (train, val)}
    result = dict(integrity="PASS", validation_replay="EXACT_MATCH", run_count=len(records),
                  total_updates=sum(r["updates"] for r in records), campaign_wall_s=campaign["wall_s"],
                  manifest=manifest, manifest_sha256=sha(campaign_root / "SHA256SUMS"),
                  protocol_sha256=sha(training.PROTOCOL), script_sha256=sha(Path(__file__)),
                  argv=sys.argv, runs=records, q_current_baseline=baseline,
                  decision="VALIDATION_TARGET_NOT_MET; NEXT_DIAGNOSTIC_REQUIRED; FINAL_NOT_CREATED",
                  final_test="NOT_CREATED", old_final_raw="NOT_READ", analysis_wall_s=time.perf_counter()-started)
    write_json(out / "audit.json", result)
    print(f"Audit PASS, {len(records)} runs; validation target NOT_MET", flush=True)


if __name__ == "__main__":
    main()
