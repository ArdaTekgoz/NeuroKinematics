"""Verify pilot and environment evidence before the unchanged full T-C00 plan."""
import argparse
import json
from pathlib import Path

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.benchmark.queries import encode_query
from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.core.contract import ROOT, load_contract, sha256
from neurokinematics.core.results import validate_result_record
from neurokinematics.core.runner import FROZEN_QUERY_PATH, SOLVER_IDS, load_queries, run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--external-worker', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    args = parser.parse_args()
    config = load_contract()
    evidence = args.evidence
    lock_path = evidence / 'environment-lock.json'
    lock = strict_json(lock_path.read_text())
    if lock['image_id'] != args.image_id:
        raise ValueError('full run image differs from audited image')
    pilot_path = evidence / 'linux-pilot/pilot-gate.json'
    pilot = strict_json(pilot_path.read_text())
    if (pilot['status'] != 'PASS' or set(pilot['solvers']) != set(SOLVER_IDS)
            or pilot['environment_lock_sha256'] != sha256(lock_path)
            or pilot['query_list_sha256'] != config['queries']['list_sha256']
            or pilot['deadline_profiles_ms'] != [10, 50] or pilot['measurement_passes'] != 1
            or pilot['selected_queries'] != 12):
        raise ValueError('pilot/environment gate mismatch')
    selection = evidence / 'linux-pilot/selection.jsonl'
    if sha256(selection) != pilot['selection_sha256']:
        raise ValueError('pilot selection changed')
    selected = [strict_json(line) for line in selection.read_text().splitlines()]
    rows, manifest = load_queries(FROZEN_QUERY_PATH, config)
    frozen = {row['query_id']: row for row in rows}
    if len(selected) != 12 or any(row != frozen.get(row['query_id']) for row in selected):
        raise ValueError('pilot selection differs from frozen queries')
    validator = CandidateValidator()
    for solver in config['solvers']:
        solver_id = solver['id']
        entry = pilot['solvers'][solver_id]
        raw_path = evidence / 'linux-pilot' / (solver_id.replace('/', '-') + '-benchmark.jsonl')
        if entry['status'] != 'PASS' or entry['record_count'] != 24 or sha256(raw_path) != entry['raw_sha256']:
            raise ValueError('pilot solver evidence changed')
        with raw_path.open('rb') as stream:
            for deadline in (10, 50):
                for query in selected:
                    payload = stream.readline()
                    record = strict_json(payload.decode())
                    if (payload != encode_query(record) or record['deadline_profile_ms'] != deadline
                            or record['measurement_pass_index'] != 0):
                        raise ValueError('pilot row order mismatch')
                    validate_result_record(record, query, solver, config, config['queries']['list_sha256'],
                                           manifest['dataset_manifest_sha256'], validator)
            if stream.readline():
                raise ValueError('extra pilot rows')
        summary = strict_json((evidence / 'linux-pilot' /
                               (solver_id.replace('/', '-') + '-benchmark-summary.json')).read_text())
        if (summary['warmup_calls'] != 20 * len(summary['worker_launch_elapsed_ns'])
                or summary['worker_start_error'] is not None):
            raise ValueError('pilot worker warmup incomplete')
    output = evidence / 'linux-full'
    if output.exists():
        raise ValueError('full output exists; preserve it before any rerun')
    output.mkdir()
    write_json(output / 'run-binding.json', {
        'image_id': args.image_id, 'environment_lock_sha256': sha256(lock_path),
        'pilot_gate_sha256': sha256(pilot_path),
        'smoke_gate_sha256': sha256(evidence / 'linux-smoke/smoke-gate.json'),
        'critical_regression_sha256': sha256(evidence / 'linux-portable-critical-regression.xml'),
        'baseline_config_sha256': sha256(ROOT / 'experiments/C1-01/baseline-config.json'),
        'runner_sha256': sha256(ROOT / 'src/neurokinematics/core/runner.py'),
        'worker_sha256': sha256(ROOT / 'src/neurokinematics/core/worker.py'),
        'launch_script_sha256': sha256(Path(__file__)),
        'expected_per_solver_records': 120000, 'expected_total_records': 600000})
    print('PILOT VERIFIED. Starting full T-C00: 600000 measured attempts.', flush=True)
    result = run('benchmark', FROZEN_QUERY_PATH, output, args.external_worker,
                 evidence / 'linux-smoke/smoke-gate.json')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'MEASURED_UNVERIFIED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
