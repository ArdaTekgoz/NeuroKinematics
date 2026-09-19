"""Execute T-F04 analytic tests and preserve measured cases and JUnit evidence."""

import argparse
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .metrics import position_error, rotation_error, quaternion_error, singularity_metrics
from .transforms import axis_angle
from .model import ROOT, load_robot
from .jacobian_validation import OUTPUT, characteristic_length, json_safe
from .validation import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = load_robot()
    junit = args.output / 'metrics-junit.xml'
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/f0_03/test_metrics.py',
               f'--junitxml={junit}', '-o', 'junit_family=legacy']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    root = ET.parse(junit).getroot()
    cases = list(root.iter('testcase'))
    failed = [c.get('name') for c in cases if c.find('failure') is not None or c.find('error') is not None or c.find('skipped') is not None]
    analytic = []
    for axis in np.eye(3):
        for angle in (0., np.pi/2, np.pi):
            q = np.r_[np.cos(angle/2), axis*np.sin(angle/2)]
            analytic.append({'axis': axis.tolist(), 'expected_rad': angle,
                             'rotation_rad': rotation_error(axis_angle(axis, angle), np.eye(3)),
                             'quaternion_rad': quaternion_error(q, [1, 0, 0, 0]),
                             'sign_pair_rad': quaternion_error(q, -q),
                             'presentation_degrees': float(np.rad2deg(angle))})
    svd = {name: singularity_metrics(np.diag(diagonal), 1.) for name, diagonal in
           {'identity': [1]*6, 'diagonal': [1, 2, 3, 4, 5, 6], 'zero': [0]*6,
            'rank_deficient': [1, 1, 1, 1, 1, 0], 'near_singular': [1, 1, 1, 1, 1, 1e-12]}.items()}
    summary = {'test_id': 'T-F04', 'status': 'PASS' if result.returncode == 0 and cases and not failed else 'FAIL',
               'test_count': len(cases), 'failed_or_skipped': failed, 'command': command,
               'exit_code': result.returncode, 'output': result.stdout + result.stderr,
               'input_hashes': inputs.hashes, 'characteristic_length_m': characteristic_length(),
               'analytic_rotations': analytic, 'singularity_cases': svd,
               'position_m': position_error([.003, .004, 0], [0, 0, 0]),
               'position_mm': 1000*position_error([.003, .004, 0], [0, 0, 0]),
               'quaternion_contract': 'wxyz; norm tolerance 1e-6; copy; sign invariant',
               'singularity_policy': 'raw SVD values; condition=Inf only for exact zero sigma_min; manipulability=product(SVD), with no clipping or epsilon; numerical rank tolerance=6*eps*sigma_max is diagnostic only',
               'test_names': [c.get('name') for c in cases]}
    write_json(args.output / 'metric-validation-summary.json', json_safe(summary))
    print(f"T-F04: {summary['status']}; {len(cases)} tests; {len(failed)} failed/skipped")
    return int(summary['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
