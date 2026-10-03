"""Capture the actual C1-04 Windows CPU package and input identities."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import platform
import sys

import numpy
import pinocchio
import torch

from neurokinematics.kinematics.model import ROOT
from neurokinematics.neural.c104 import EVIDENCE, sha, write_json


def main() -> None:
    torch.set_num_threads(1)
    record = {"task": "C1-04", "utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "python": sys.version,
              "python_executable": sys.executable, "torch": torch.__version__,
              "numpy": numpy.__version__, "pinocchio": pinocchio.__version__,
              "torch_num_threads": torch.get_num_threads(),
              "thread_environment": {key: os.environ.get(key) for key in
                                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "torch_cuda": "NOT_RUN", "Linux": "NOT_RUN", "GPU_training": "NOT_USED",
              "pixi_lock_sha256": sha(ROOT / "pixi.lock"),
              "torch_lock_sha256": sha(ROOT / "experiments/C1-03/requirements-win-cpu.lock"),
              "runtime_supplement_sha256": sha(ROOT / "experiments/C1-03/stage2/runtime-supplement.lock"),
              "config_sha256": sha(ROOT / "experiments/C1-04/config.json"),
              "neural_source_sha256": sha(ROOT / "src/neurokinematics/neural/c104.py")}
    write_json(EVIDENCE / "stage2/environment.json", record)
    print(json.dumps({"status": "RECORDED", "torch": record["torch"], "python": sys.version.split()[0],
                      "neural_source_sha256": record["neural_source_sha256"]}))


if __name__ == "__main__":
    main()
