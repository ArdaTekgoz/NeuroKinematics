"""Quantify C1-04 pose-only label ambiguity on train/validation roots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurokinematics.neural.c104 import EVIDENCE, load_data, write_json


def summarize(rows) -> dict:
    pairs = {}
    for index, source in enumerate(rows.source_sample_id):
        pairs.setdefault(str(source), {})[str(rows.mode[index])] = index
    differences = []
    rad_differences = []
    for root in pairs.values():
        if set(root) != {"local", "wide"}:
            raise ValueError("missing pair mode")
        a, b = root["local"], root["wide"]
        if not np.array_equal(rows.pose_only[a], rows.pose_only[b]):
            raise ValueError("source pose differs across pair modes")
        if rows.label_present[b]:
            differences.append(np.linalg.norm(rows.target_normalized[a] - rows.target_normalized[b]))
            rad_differences.append(np.linalg.norm(rows.q_target[a] - rows.q_target[b]))
    d = np.asarray(differences)
    q = np.asarray(rad_differences)
    return {"split": rows.split, "roots": len(pairs), "paired_labeled_roots": len(d),
            "pose_only_identical_feature_pairs": len(pairs),
            "normalized_label_l2_median": float(np.median(d)),
            "normalized_label_l2_p95": float(np.percentile(d, 95)),
            "q_target_l2_rad_median": float(np.median(q)),
            "q_target_l2_rad_p95": float(np.percentile(q, 95)),
            "pose_only_supervised_mse_lower_bound": float(np.sum(d ** 2) / (2 * int(rows.label_present.sum()))),
            "bound_scope": "exact duplicate pose features with two distinct normalized q labels; ignores other approximation error"}


def main() -> None:
    train, validation = load_data(label_fk=False)
    result = {"status": "MEASURED", "test_and_benchmark": "SEALED_NOT_RUN",
              "train": summarize(train), "validation": summarize(validation)}
    write_json(EVIDENCE / "stage2/ambiguity-diagnostic.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
