"""C1-04 Stage 2 entry point; test and benchmark splits are sealed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"

from neurokinematics.kinematics.model import ROOT
from neurokinematics.neural.c104 import (CONFIG, EVIDENCE, VARIANTS, contracts,
    controlled_subset, evaluate, load_checkpoint, load_data, predict, read_json,
    reject_shifted_labels, sha, train_pair, write_json)


STAGE2 = EVIDENCE / "stage2"


def preflight(tag: str = "") -> tuple:
    subprocess.run([sys.executable, str(ROOT / "scripts/check_c104_stage1.py"), "--check"], cwd=ROOT, check=True)
    config, _, _ = contracts()
    if platform.system() != "Windows" or platform.machine() not in ("AMD64", "x86_64"):
        raise RuntimeError("C1-04 frozen runtime is Windows x64 CPU")
    import torch
    if torch.__version__ != "2.10.0+cpu":
        raise RuntimeError("Torch runtime drift")
    train, validation = load_data(label_fk=True)
    pilot_train, pilot_validation = controlled_subset(train, validation)
    negative = reject_shifted_labels(pilot_train)
    result = {"status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
              "stage1_commit": "9b6029bf62feb2caff3b173c600868acc9d57c0c",
              "config_sha256": sha(CONFIG), "torch": torch.__version__,
              "python": sys.version, "platform": platform.platform(),
              "train_rows": len(train.pair_id), "validation_rows": len(validation.pair_id),
              "train_labels": int(train.label_present.sum()),
              "validation_labels": int(validation.label_present.sum()),
              "pilot_train_pair_ids": pilot_train.pair_id.tolist(),
              "pilot_validation_pair_ids": pilot_validation.pair_id.tolist(),
              "shifted_label_negative": negative,
              "test_or_benchmark_opened": False}
    write_json(STAGE2 / (f"preflight-{tag}.json" if tag else "preflight.json"), result)
    return train, validation


def run_pilot() -> None:
    train, validation = preflight("pilot")
    pt, pv = controlled_subset(train, validation)
    result = train_pair(pt, pv, read_json(CONFIG)["training"]["seeds"][0], pilot=True,
                        evidence=STAGE2, weights=ROOT / "data/generated/C1-04/pilot")
    if result["t_c03_status"] != "PASS":
        raise RuntimeError("T-C03 pilot failed; full training gate closed")
    print(json.dumps({"T-C03": result["t_c03_status"], "gates": result["t_c03_gates"],
                      "epochs": result["epochs"], "wall_s": result["elapsed_wall_s"]}))


def run_full(seed: int) -> None:
    config = read_json(CONFIG)
    if seed not in config["training"]["seeds"]:
        raise ValueError("seed outside frozen E-C01 list")
    pilot = read_json(STAGE2 / "pilot-summary.json")
    if pilot["t_c03_status"] != "PASS":
        raise ValueError("T-C03 gate not passed")
    train, validation = preflight(f"seed-{seed}")
    weights = ROOT / config["resources"]["local_weight_root"] / f"seed-{seed}"
    result = train_pair(train, validation, seed, pilot=False, evidence=STAGE2, weights=weights)
    evaluated = {}
    for variant in VARIANTS:
        checkpoint = Path(result["best_checkpoints"][variant]["path"])
        model, metadata = load_checkpoint(checkpoint, expected_variant=variant)
        evaluated[variant] = evaluate(model, metadata, validation,
                                      output=STAGE2 / f"seed-{seed}-{variant}-validation.jsonl")
    write_json(STAGE2 / f"seed-{seed}-evaluation.json", {"status": "COMPLETE", "seed": seed,
                                                       "variants": evaluated})
    print(json.dumps({"seed": seed, "epochs": result["epochs"],
                      "best_epoch": result["best_epoch"],
                      "validation_loss": result["best_validation_loss"],
                      "valid_raw": {v: e["counts"].get("valid_raw", 0) for v, e in evaluated.items()}}))


def audit() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/check_c104_stage1.py"), "--check"], cwd=ROOT, check=True)
    config = read_json(CONFIG)
    pilot = read_json(STAGE2 / "pilot-summary.json")
    if pilot["t_c03_status"] != "PASS":
        raise ValueError("T-C03 failed")
    results = []
    for seed in config["training"]["seeds"]:
        run = read_json(STAGE2 / f"seed-{seed}-summary.json")
        evaluation = read_json(STAGE2 / f"seed-{seed}-evaluation.json")
        if run["status"] != "COMPLETE" or evaluation["status"] != "COMPLETE":
            raise ValueError("incomplete seed")
        if run["optimizer_steps_per_model"] <= 0 or run["epochs"] <= 0:
            raise ValueError("missing training budget")
        for variant in VARIANTS:
            checkpoint = run["best_checkpoints"][variant]
            path = Path(checkpoint["path"])
            if not path.is_file() or path.stat().st_size != checkpoint["bytes"] or sha(path) != checkpoint["sha256"]:
                raise ValueError("checkpoint SHA/size/access drift")
            load_checkpoint(path, expected_variant=variant)
            log = run["epoch_logs"][variant]
            if sha(Path(log["path"])) != log["sha256"]:
                raise ValueError("epoch log SHA drift")
            detail = evaluation["variants"][variant]
            if detail["counts"]["rows"] != 3600 or sha(Path(detail["rows_path"])) != detail["rows_sha256"]:
                raise ValueError("validation row evidence drift")
            results.append({"seed": seed, "variant": variant, "epoch": checkpoint["epoch"],
                            "checkpoint_sha256": checkpoint["sha256"],
                            "validation_loss": run["best_validation_loss"][variant],
                            "valid_raw": detail["counts"].get("valid_raw", 0),
                            "out_of_limits": detail["counts"].get("out_of_limits", 0),
                            "nonfinite": detail["counts"].get("nonfinite", 0),
                            "position_m": detail["breakdowns"]["overall"]["position_m"],
                            "orientation_deg": detail["breakdowns"]["overall"]["orientation_deg"]})
    output = {"status": "E_C01_EVIDENCE_COMPLETE", "T-C03": "PASS", "E-C01": "3_PAIRED_SEEDS_COMPLETE",
              "config_sha256": sha(CONFIG), "results": results,
              "test_and_benchmark": "SEALED_NOT_RUN", "Linux": "NOT_RUN", "CUDA": "NOT_RUN"}
    write_json(STAGE2 / "audit.json", output)
    print(json.dumps(output))


def inference(checkpoint: Path, payload_path: Path) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/check_c104_stage1.py"), "--check"], cwd=ROOT, check=True)
    model, metadata = load_checkpoint(checkpoint)
    value = read_json(payload_path)
    result = predict(model, metadata, value["position_m"], value["quaternion_wxyz"], value.get("q_current"))
    print(json.dumps({"checkpoint_sha256": sha(checkpoint), "metadata": metadata, "prediction": result}, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "pilot", "full", "audit", "infer"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
        print("PREFLIGHT PASS")
    elif args.stage == "pilot":
        run_pilot()
    elif args.stage == "full":
        if args.seed is None:
            parser.error("full requires --seed")
        run_full(args.seed)
    elif args.stage == "audit":
        audit()
    else:
        if args.checkpoint is None or args.input is None:
            parser.error("infer requires --checkpoint and --input")
        inference(args.checkpoint, args.input)


if __name__ == "__main__":
    main()
