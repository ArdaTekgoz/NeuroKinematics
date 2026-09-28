"""Verify completed methods of an incomplete run without modifying raw files."""
import json
from pathlib import Path

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.core.contract import sha256
from neurokinematics.core.runner import FROZEN_QUERY_PATH, SOLVER_IDS, summarize_file, write_json


def main():
    output = Path('/evidence/linux-full')
    manifest = strict_json((output / 'benchmark-manifest.json').read_text())
    result = {'task': 'C1-01', 'status': 'PARTIAL_VERIFIED', 'full_acceptance': False, 'solvers': {}}
    for solver_id in SOLVER_IDS:
        entry = manifest['solvers'][solver_id]
        if entry['status'] != 'MEASURED_UNVERIFIED' or entry['record_count'] != 120000:
            result['solvers'][solver_id] = {'status': 'INCOMPLETE', 'record_count': entry['record_count']}
            continue
        safe = solver_id.replace('/', '-')
        raw = output / (safe + '-benchmark.jsonl')
        if sha256(raw) != entry['raw_sha256']:
            raise ValueError(f'raw hash changed: {solver_id}')
        summary = summarize_file(raw, solver_id, 'benchmark', FROZEN_QUERY_PATH)
        write_json(output / (safe + '-verified-summary.json'), summary)
        result['solvers'][solver_id] = {'status': summary['status'],
                                        'record_count': summary['record_count'],
                                        'raw_sha256': summary['raw_sha256'],
                                        'common_status_counts': summary['groups']['all']['common_status_counts']}
        print('Verified ' + solver_id, flush=True)
    write_json(output / 'completed-methods-verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
