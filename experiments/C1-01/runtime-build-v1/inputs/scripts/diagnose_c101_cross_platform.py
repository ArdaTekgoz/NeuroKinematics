"""Read-only diagnosis of archived F0-06 FK residual portability."""
import json
import platform
from pathlib import Path

import numpy as np
import pinocchio

from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.solvers.dls import SolverResult, SolverStatus


def main():
    path = Path('/work/experiments/F0-06/reproduction/a/results.jsonl')
    if not path.exists():
        path = Path('experiments/F0-06/reproduction/a/results.jsonl')
    row = json.loads(path.read_bytes().splitlines()[0])
    result = SolverResult(SolverStatus(row['solver_status']), row['termination_reason'],
                          np.asarray(row['q_candidate'], dtype=np.float64), row['iterations'],
                          row['first_profile_a_iteration'], row['first_profile_b_iteration'],
                          row['solve_elapsed_ns'])
    verdict = CandidateValidator().validate(result, row['target_position_m'], row['target_quaternion_wxyz'])
    fields = ('position_error_m', 'orientation_error_rad', 'orientation_error_deg',
              'profile_a_geometry', 'profile_b_geometry', 'joint_limits')
    print(json.dumps({'platform': platform.platform(), 'numpy': np.__version__,
                      'pinocchio': pinocchio.__version__,
                      'values': {key: {'archived': row[key], 'recomputed': getattr(verdict, key)}
                                 for key in fields}}, indent=2))


if __name__ == '__main__':
    main()
