"""Disclosed 10/50 ms subset pilot; never substitutes for full T-C00."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.benchmark.queries import encode_query
from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.core.contract import load_contract, sha256
from neurokinematics.core.results import validate_result_record
from neurokinematics.core.runner import (FROZEN_QUERY_PATH, SOLVER_IDS, load_queries,
                                        run_method, smoke_gate, utc_now, write_json)


def select_rows(rows):
    # Two first frozen queries for each subset/start-class pair, in original order.
    counts = {}
    selected = []
    for row in rows:
        key = (row['subset'], row['start_class'])
        if counts.get(key, 0) < 2:
            selected.append(row)
            counts[key] = counts.get(key, 0) + 1
    expected = {(subset, start) for subset in ('main', 'boundary', 'singularity')
                for start in ('local', 'wide')}
    if set(counts) != expected or any(n != 2 for n in counts.values()):
        raise ValueError('pilot requires two queries from each of six frozen groups')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--external-worker', type=Path, required=True)
    args = parser.parse_args()
    config = load_contract()
    release = Path('/etc/os-release').read_text()
    if (sys.platform != 'linux' or platform.machine() != 'x86_64'
            or 'ID=ubuntu' not in release or 'VERSION_ID="24.04"' not in release
            or os.environ.get('ROS_DISTRO') != 'jazzy' or sorted(os.sched_getaffinity(0)) != [0, 1]):
        raise ValueError('pinned Ubuntu x86_64 Jazzy / CPU 0,1 required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) not in ('1', '2'):
            raise ValueError(f'invalid thread policy: {name}')
    suites = ET.parse(args.evidence / 'linux-portable-critical-regression.xml').getroot().findall('testsuite')
    if (sum(int(s.attrib['tests']) for s in suites) != 211
            or any(int(s.attrib[k]) for s in suites for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('211-test scoped Linux regression evidence required')
    smoke_gate(args.evidence / 'linux-smoke/smoke-gate.json', config, FROZEN_QUERY_PATH)
    output = args.evidence / 'linux-pilot'
    if output.exists():
        raise ValueError('pilot directory exists; preserve it and use a new evidence root for rerun')
    output.mkdir()
    rows, manifest = load_queries(FROZEN_QUERY_PATH, config)
    selected = select_rows(rows)
    selection = output / 'selection.jsonl'
    selection.write_bytes(b''.join(encode_query(row) for row in selected))
    # A separately disclosed measurement plan; original config and full runner unchanged.
    plan = copy.deepcopy(config)
    plan['benchmark']['measurement_passes'] = 1
    result = {'task': 'C1-01', 'mode': 'pilot', 'started_utc': utc_now(),
              'status': 'IN_PROGRESS', 'query_list_sha256': sha256(FROZEN_QUERY_PATH),
              'selection_sha256': sha256(selection), 'selected_queries': len(selected),
              'deadline_profiles_ms': [10, 50], 'measurement_passes': 1,
              'environment_lock_sha256': sha256(args.evidence / 'environment-lock.json'),
              'scope': 'subset integration pilot; not full T-C00', 'solvers': {}}
    write_json(output / 'pilot-gate.json', result)
    for solver_id in SOLVER_IDS:
        solver = next(s for s in config['solvers'] if s['id'] == solver_id)
        summary = run_method(solver, plan, selected, manifest, output, args.external_worker,
                             rows[:20], mode='benchmark')
        raw = Path(summary['raw_path'])
        validator = CandidateValidator()
        with raw.open('rb') as stream:
            for deadline in config['benchmark']['deadline_profiles_ms']:
                for query in selected:
                    payload = stream.readline()
                    row = strict_json(payload.decode())
                    if (encode_query(row) != payload or row['deadline_profile_ms'] != deadline
                            or row['measurement_pass_index'] != 0):
                        raise ValueError('pilot ordering/encoding mismatch')
                    validate_result_record(row, query, solver, config, config['queries']['list_sha256'],
                                           manifest['dataset_manifest_sha256'], validator)
            if stream.readline():
                raise ValueError('extra pilot rows')
        fatal = sum(summary['status_counts'].get(k, 0) for k in
                    ('PROCESS_FAILURE', 'INSTALLATION_FAILURE', 'ADAPTER_ERROR',
                     'INVALID_OUTPUT', 'VALIDATION_ERROR'))
        passed = summary['record_count'] == 24 and summary['worker_start_error'] is None and fatal == 0
        result['solvers'][solver_id] = {'status': 'PASS' if passed else 'FAIL',
                                        'record_count': summary['record_count'],
                                        'raw_sha256': sha256(raw), 'status_counts': summary['status_counts']}
        write_json(output / 'pilot-gate.json', result)
    result['status'] = 'PASS' if all(s['status'] == 'PASS' for s in result['solvers'].values()) else 'FAIL'
    result['finished_utc'] = utc_now()
    write_json(output / 'pilot-gate.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
