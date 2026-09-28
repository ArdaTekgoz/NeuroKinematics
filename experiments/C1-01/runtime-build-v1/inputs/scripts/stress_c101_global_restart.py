"""Global worker restart stress on 250 frozen queries at each deadline."""
import argparse
import copy
import json
import os
import platform
from pathlib import Path
import xml.etree.ElementTree as ET

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.benchmark.queries import encode_query
from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.core.contract import load_contract, sha256
from neurokinematics.core.results import validate_result_record
from neurokinematics.core.runner import FROZEN_QUERY_PATH, load_queries, run_method, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--external-worker', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (platform.system() != 'Linux' or platform.machine() != 'x86_64'
            or sorted(os.sched_getaffinity(0)) != [0, 1]
            or os.environ.get('ROS_DISTRO') != 'jazzy'):
        raise ValueError('Linux x86_64 Jazzy CPU0/1 required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) not in ('1', '2'):
            raise ValueError('thread policy mismatch')
    evidence = args.evidence
    suites = ET.parse(evidence / 'linux-portable-critical-regression.xml').getroot().findall('testsuite')
    if (sum(int(s.attrib['tests']) for s in suites) != 211
            or any(int(s.attrib[k]) for s in suites for k in ('errors', 'failures', 'skipped'))):
        raise ValueError('211-test scoped Linux regression must pass first')
    config = load_contract()
    rows, manifest = load_queries(FROZEN_QUERY_PATH, config)
    selected = rows[:250]
    solver = next(s for s in config['solvers'] if s['id'] == 'pick_ik/global')
    output = args.output or evidence / 'linux-global-restart-stress'
    if output.exists():
        raise ValueError('stress evidence exists; preserve it before rerun')
    output.mkdir()
    shared_memory_before = sorted(p.name for p in Path('/dev/shm').iterdir())
    plan = copy.deepcopy(config)
    plan['benchmark']['measurement_passes'] = 1
    summary = run_method(solver, plan, selected, manifest, output, args.external_worker,
                         rows[:20], mode='benchmark')
    raw = Path(summary['raw_path'])
    result = {'task': 'C1-01', 'mode': 'restart_stress', 'status': 'INCOMPLETE',
              'scope': 'transport stress; not full T-C00 or performance comparison',
              'record_count': summary['record_count'], 'expected_record_count': 500,
              'worker_start_error': summary['worker_start_error'],
              'restart_warmups': summary['restart_warmups'],
              'warmup_calls': summary['warmup_calls'], 'status_counts': summary['status_counts'],
              'raw_sha256': sha256(raw),
              'environment_lock_sha256': sha256(evidence / 'environment-lock.json')}
    result['transport_environment'] = {
        name: os.environ.get(name) for name in ('RMW_IMPLEMENTATION', 'FASTDDS_BUILTIN_TRANSPORTS')}
    result['shared_memory_entries_before'] = shared_memory_before
    result['shared_memory_entries_after'] = sorted(p.name for p in Path('/dev/shm').iterdir())
    if summary['record_count'] == 500 and summary['worker_start_error'] is None:
        validator = CandidateValidator()
        with raw.open('rb') as stream:
            for deadline in (10, 50):
                for query in selected:
                    payload = stream.readline()
                    record = strict_json(payload.decode())
                    if (encode_query(record) != payload or record['deadline_profile_ms'] != deadline
                            or record['measurement_pass_index'] != 0):
                        raise ValueError('stress order/encoding mismatch')
                    validate_result_record(record, query, solver, config, config['queries']['list_sha256'],
                                           manifest['dataset_manifest_sha256'], validator)
            if stream.readline():
                raise ValueError('extra stress records')
        fatal = sum(summary['status_counts'].get(k, 0) for k in
                    ('PROCESS_FAILURE', 'INSTALLATION_FAILURE', 'INVALID_OUTPUT',
                     'ADAPTER_ERROR', 'VALIDATION_ERROR'))
        warm = summary['warmup_calls'] == 20 * len(summary['worker_launch_elapsed_ns'])
        result['status'] = 'PASS' if fatal == 0 and warm and summary['restart_warmups'] >= 100 else 'FAIL'
    write_json(output / 'stress-gate.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
