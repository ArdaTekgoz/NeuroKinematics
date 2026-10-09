"""User-run validation campaign; smoke mode cannot open the full campaign."""
import argparse
import copy
from pathlib import Path
import time
import torch
from neurokinematics.neural import c106r, c106r_training as training
from neurokinematics.neural.c104 import ROOT, read_json, write_json, load_data, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    c106r.configure()
    spec = read_json(training.PROTOCOL)
    if args.smoke:
        train, val = load_data(label_fk=True)
        results = []
        for arm in spec["arms"]:
            smoke = copy.deepcopy(spec)
            smoke.update(epochs=2, validation_every_epochs=1)
            contract = dict(purpose="SMOKE_NOT_MAIN_TRAINING", arm=arm, production_protocol_sha256=sha(training.PROTOCOL))
            path = ROOT/"data/generated/C1-06R/smoke-round1"/arm
            torch.cuda.reset_peak_memory_stats()
            result = training.train_run(train, val, smoke, arm, spec["seed_list"][0], path, contract)
            results.append(dict(arm=arm, epochs=2, updates=sum(x["updates"] for x in result["history"]),
                                wall_s=result["this_invocation_wall_s"], epoch_train_seconds=[x["wall_s"] for x in result["history"]],
                                peak_cuda_bytes=torch.cuda.max_memory_allocated(), initial_hash=result["initial_hash"],
                                best_checkpoint=result["best_checkpoint"], final_state_hash=result["final_state_hash"]))
        if len(set(r["initial_hash"] for r in results)) != 1:
            raise ValueError("paired initial states differ")
        write_json(ROOT/"experiments/C1-06R/training-smoke.json", dict(status="PASS", results=results,
                   main_training="NOT_RUN", protocol_sha256=sha(training.PROTOCOL), code_sha256=sha(Path(training.__file__))))
        return
    from check_c106r_training import check
    check()
    started = time.perf_counter()
    summaries = []
    for seed in spec["seed_list"]:
        for arm in spec["arms"]:
            result = training.production(arm, seed, resume=args.resume)
            summaries.append(dict(arm=arm, seed=seed, status=result["status"], best=result["best"], best_checkpoint=result["best_checkpoint"]))
    out = ROOT/"data/generated/C1-06R/round1"
    write_json(out/"campaign-complete.json", dict(status="COMPLETE_VALIDATION_ONLY", runs=summaries,
               wall_s=time.perf_counter()-started, final_test="NOT_CREATED", protocol_sha256=sha(training.PROTOCOL)))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS",) and not p.name.endswith(".tmp"))
    (out/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(out).as_posix()}\n" for p in files), encoding="utf-8")
    print("Completed validation campaign. Output: " + str(out), flush=True)


if __name__ == "__main__":
    main()
