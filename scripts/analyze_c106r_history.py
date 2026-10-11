"""Diagnose historical train/validation logs; never read final-test queries."""
from pathlib import Path
import numpy as np
from neurokinematics.neural.c104 import ROOT, read_json, write_json, sha


def main():
    output = ROOT / "experiments/C1-06R/r0r1/history-analysis.json"
    if output.exists():
        raise FileExistsError(output)
    records = []
    for path in sorted((ROOT / "experiments/C1-05/stage2").glob("E-C*/seed-*/attempt-001/*-epochs.jsonl")):
        import json
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        qbest = min(rows, key=lambda r: r["validation"]["components"]["q"])
        posebest = min(rows, key=lambda r: r["validation"]["components"]["p"] + r["validation"]["components"]["R"])
        last = rows[-1]
        def extract(row):
            return dict(epoch=row["epoch"], train=row["train"]["components"], validation=row["validation"]["components"])
        tanh = [r["gradients"]["tanh"]["saturated_fraction_per_joint"] for r in rows if "tanh" in r["gradients"]]
        selected_pose = qbest["validation"]["components"]["p"] + qbest["validation"]["components"]["R"]
        minimum_pose = posebest["validation"]["components"]["p"] + posebest["validation"]["components"]["R"]
        records.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                            experiment=path.parts[-4], seed=last["seed"], variant=last["variant"],
                            epochs=len(rows), optimizer_steps=last["optimizer_steps"],
                            q_selected=extract(qbest), pose_proxy_minimum=extract(posebest), last=extract(last),
                            pose_proxy_relative_reduction_if_selected=(selected_pose-minimum_pose)/selected_pose,
                            maximum_logged_tanh_saturation=float(np.max(tanh)) if tanh else None))
    write_json(output, dict(status="ANALYZED", records=records,
                           scope="Logged labeled train/validation losses; pose proxy p+R is NOT Profile A success",
                           interpretation="Selection disagreement and saturation are diagnostics, not proven causes of zero final success",
                           final_raw="NOT_READ", checkpoint_reselection="NOT_PERFORMED"))
    for r in records:
        if r["variant"] == "FK_TANH":
            print(r["seed"], r["epochs"], r["q_selected"]["epoch"], r["pose_proxy_minimum"]["epoch"], r["pose_proxy_relative_reduction_if_selected"], r["maximum_logged_tanh_saturation"])


if __name__ == "__main__":
    main()
