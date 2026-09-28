"""Capture malformed IPC lines during global restarts without accepting them."""
import copy
import json
from pathlib import Path

from neurokinematics.core import worker
from neurokinematics.core.contract import load_contract, sha256
from neurokinematics.core.runner import FROZEN_QUERY_PATH, load_queries, run_method, write_json


def main():
    output = Path('/evidence/linux-ready-diagnostic')
    if output.exists():
        raise ValueError('diagnostic directory exists; preserve before rerun')
    output.mkdir()
    original_parser = worker.strict_json
    failures = []

    def capture(raw):
        try:
            return original_parser(raw)
        except ValueError:
            failure = {'raw_text': raw, 'raw_repr': repr(raw),
                       'utf8_hex': raw.encode('utf-8').hex(),
                       'shared_memory_entries': sorted(p.name for p in Path('/dev/shm').iterdir())}
            failures.append(failure)
            write_json(output / 'invalid-protocol-lines.json', failures)
            raise

    worker.strict_json = capture
    config = load_contract()
    rows, manifest = load_queries(FROZEN_QUERY_PATH, config)
    plan = copy.deepcopy(config)
    plan['benchmark']['measurement_passes'] = 1
    plan['benchmark']['deadline_profiles_ms'] = [10]
    solver = next(s for s in config['solvers'] if s['id'] == 'pick_ik/global')
    executable = Path('/opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker')
    try:
        summary = run_method(solver, plan, rows[:250], manifest, output, executable,
                             rows[:20], mode='benchmark')
    finally:
        worker.strict_json = original_parser
    report = {'scope': 'IPC diagnostic; not full benchmark',
              'record_count': summary['record_count'], 'expected_record_count': 250,
              'worker_start_error': summary['worker_start_error'],
              'invalid_protocol_lines': failures,
              'environment_lock_sha256': sha256(Path('/evidence/environment-lock.json'))}
    write_json(output / 'diagnostic-report.json', report)
    display = {**report, 'invalid_protocol_lines': [
        {'raw_text': item['raw_text'], 'utf8_hex': item['utf8_hex'],
         'shared_memory_entry_count': len(item['shared_memory_entries'])} for item in failures]}
    print(json.dumps(display, indent=2), flush=True)


if __name__ == '__main__':
    main()
