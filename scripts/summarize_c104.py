"""Summarize paired C1-04 validation rows; never open test or benchmark."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from neurokinematics.kinematics.model import ROOT
from neurokinematics.neural.c104 import CONFIG, read_json, sha, write_json


STAGE2 = ROOT / "experiments/C1-04/stage2"


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p95": None, "p99": None}
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}


def rows(path: Path) -> list[dict]:
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(result) != 3600 or len({row["pair_id"] for row in result}) != 3600:
        raise ValueError("validation inventory mismatch")
    return result


def grouped(data: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in data:
        keys = ("overall", f"mode/{row['mode']}", f"family/{row['family']}",
                f"label/{row['label_present']}",
                f"family/{row['family']}/mode/{row['mode']}/label/{row['label_present']}")
        for key in keys:
            groups[key].append(row)
    output = {}
    for key, part in sorted(groups.items()):
        output[key] = {"n": len(part), "labeled": sum(row["label_present"] for row in part),
                       "valid_raw": sum(row["in_limits"] for row in part),
                       "out_of_limits": sum(row["finite"] and not row["in_limits"] for row in part),
                       "nonfinite": sum(not row["finite"] for row in part),
                       "profile_a_success": sum(row["in_limits"] and row["position_error_m"] <= 0.002 and row["orientation_error_deg"] <= 1.0 for row in part),
                       "profile_b_success": sum(row["in_limits"] and row["position_error_m"] <= 0.001 and row["orientation_error_deg"] <= 0.5 for row in part),
                       "position_m": stats([row["position_error_m"] for row in part if row["position_error_m"] is not None]),
                       "orientation_deg": stats([row["orientation_error_deg"] for row in part if row["orientation_error_deg"] is not None]),
                       "q_mae_rad": stats([row["q_mae_rad"] for row in part if row["q_mae_rad"] is not None]),
                       "q_l2_rad": stats([row["q_l2_rad"] for row in part if row["q_l2_rad"] is not None])}
    return output


def main() -> None:
    config = read_json(CONFIG)
    output = {"status": "COMPLETE", "split": "validation", "test_and_benchmark": "SEALED_NOT_RUN",
              "config_sha256": sha(CONFIG), "seeds": []}
    lines = ["# E-C01 eşli validation özeti", "", "Test ve 10.000 sorguluk benchmark açılmadı. FK metrikleri yalnız limit içi ham q için Pinocchio referansıyla ölçüldü; limit ihlalleri ayrıca sayıldı.", "",
             "| Seed | Model | En iyi epoch | Val q loss | N geçerli / 3.600 | Limit dışı | Profil A başarı / 3.600 | Konum medyan/P95 (m) | Yönelim medyan/P95 (°) | q MAE medyan (rad) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in config["training"]["seeds"]:
        run = read_json(STAGE2 / f"seed-{seed}-summary.json")
        if run["status"] != "COMPLETE":
            raise ValueError("incomplete seed")
        pair = {"seed": seed, "epochs": run["epochs"], "optimizer_steps_per_model": run["optimizer_steps_per_model"],
                "wall_s": run["elapsed_wall_s"], "peak_rss_bytes_observed": run["peak_rss_bytes_observed"],
                "models": {}}
        model_rows = {}
        for variant in ("pose_only", "conditioned"):
            path = STAGE2 / f"seed-{seed}-{variant}-validation.jsonl"
            data = rows(path)
            model_rows[variant] = data
            detail = grouped(data)
            pair["models"][variant] = {"parameter_count": config["models"]["parameter_count"][variant],
                                       "best_epoch": run["best_epoch"][variant],
                                       "validation_q_loss": run["best_validation_loss"][variant],
                                       "checkpoint": run["best_checkpoints"][variant],
                                       "validation_rows_sha256": sha(path), "breakdowns": detail}
            overall = detail["overall"]
            pos, ori, qmae = overall["position_m"], overall["orientation_deg"], overall["q_mae_rad"]
            lines.append(f"| {seed} | {variant} | {run['best_epoch'][variant]} | {run['best_validation_loss'][variant]:.6g} | {overall['valid_raw']} | {overall['out_of_limits']} | {overall['profile_a_success']} | {pos['median']:.4g} / {pos['p95']:.4g} | {ori['median']:.4g} / {ori['p95']:.4g} | {qmae['median']:.4g} |")
        a, b = model_rows["pose_only"], model_rows["conditioned"]
        if [r["pair_id"] for r in a] != [r["pair_id"] for r in b]:
            raise ValueError("model evaluation rows not paired")
        paired = [(ra, rb) for ra, rb in zip(a, b) if ra["in_limits"] and rb["in_limits"]]
        delta_p = [rb["position_error_m"] - ra["position_error_m"] for ra, rb in paired]
        delta_r = [rb["orientation_error_deg"] - ra["orientation_error_deg"] for ra, rb in paired]
        pair["paired_valid_both"] = len(paired)
        pair["conditioned_minus_pose_position_m"] = stats(delta_p)
        pair["conditioned_minus_pose_orientation_deg"] = stats(delta_r)
        pair["conditioned_position_win_fraction_valid_both"] = sum(x < 0 for x in delta_p) / len(delta_p) if delta_p else None
        output["seeds"].append(pair)
    lines += ["", "Eksi fark conditioned lehinedir; yalnız iki ham q da limit içindeyken eşli FK farkı hesaplanır. Bu seçim yanlılığı yaratabileceği için bütün 3.600 satırın geçerlilik sayıları tabloda korunur.", "",
              "| Seed | İki model de geçerli N | Conditioned − pose konum farkı medyan (m) | Conditioned − pose yönelim farkı medyan (°) | Conditioned konum kazanım oranı |",
              "|---:|---:|---:|---:|---:|"]
    for pair in output["seeds"]:
        lines.append(f"| {pair['seed']} | {pair['paired_valid_both']} | {pair['conditioned_minus_pose_position_m']['median']:.4g} | {pair['conditioned_minus_pose_orientation_deg']['median']:.4g} | {pair['conditioned_position_win_fraction_valid_both']:.3f} |")
    write_json(STAGE2 / "E-C01-summary.json", output)
    (STAGE2 / "E-C01-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "COMPLETE", "seeds": len(output["seeds"]), "rows_per_model_seed": 3600}))


if __name__ == "__main__":
    main()
