"""Stage 1: inspect manifests, hash/count sealed bytes, never decode final rows."""
import hashlib
import argparse
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/C1-06'


def read(path):
    return json.loads((ROOT / path).read_bytes())


def fingerprint(path, count_lines=False):
    h = hashlib.sha256()
    size = lines = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
            size += len(block)
            if count_lines:
                lines += block.count(b'\n')
    return dict(sha256=h.hexdigest(), bytes=size, **({'rows': lines} if count_lines else {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT / 'input-hashes.json')
    args = parser.parse_args()
    files = {}
    errors = []

    def check(path, expected=None, size=None, rows=None):
        p = ROOT / path
        relative = p.relative_to(ROOT).as_posix()
        item = dict(path=relative, accessible=p.is_file(),
                    storage='LOCAL_ONLY' if relative.startswith('data/generated/') or
                    (relative.startswith('experiments/C1-01/') and relative.endswith('benchmark.jsonl')) else 'GIT',
                    expected_sha256=expected)
        if item['accessible']:
            item.update(fingerprint(p, rows is not None))
            item['matches'] = ((expected is None or item['sha256'] == expected) and
                               (size is None or item['bytes'] == size) and
                               (rows is None or item['rows'] == rows))
        else:
            item['matches'] = False
        if not item['matches']:
            errors.append(relative)
        files[relative] = item
        return item

    handoff = read('experiments/C1-05/stage2/C1-06-handoff.json')
    for key in ('config', 'input_manifest', 'dataset_manifest', 'normalization'):
        check(handoff[key + '_path'] if key != 'config' else handoff['frozen_config_path'],
              handoff[key + '_sha256'])
    # Full historical input manifest anchors robot, TCP, data and environment.
    for item in read('experiments/C1-05/input-hashes.json')['files']:
        check(item['path'], item['sha256'], item['bytes'])
    frozen = read('experiments/C1-01/frozen-hashes.json')
    for path, digest in frozen['files'].items():
        check(path, digest, rows=12000 if path.endswith('query-list.jsonl') else None)
    gate = read('experiments/C1-01/udp-v2/full/gate.json')
    verify = read('experiments/C1-01/udp-v2/verify/gate.json')
    check('experiments/C1-01/udp-v2/full/gate.json', verify['full_gate_sha256'])
    check('experiments/C1-01/udp-v2/runtime-lock.json', gate['runtime_sha256'])
    check('experiments/C1-01/udp-v2/verify/gate.json')
    if verify['status'] != 'PASS' or gate['query_list_sha256'] != frozen['query_list_sha256']:
        errors.append('baseline acceptance/query binding')
    baseline = []
    for method, item in gate['solvers'].items():
        result = check('experiments/C1-01/udp-v2/full/' + item['raw_file'],
                       item['raw_sha256'], item['raw_bytes'], item['record_count'])
        baseline.append(dict(method=method, **result))
    for path, digest in verify['files'].items():
        check('experiments/C1-01/udp-v2/verify/' + path, digest)
    manifest = read('experiments/C1-02/dataset-manifest.json')
    for shard in manifest['shards']:
        check('data/generated/C1-02/v1/' + shard['path'], shard['file_sha256'])
    checkpoints = []
    audit = read('experiments/C1-05/stage2/results-audit.json')
    for item in audit['checkpoints']:
        if item['kind'] != 'best_checkpoints':
            continue
        result = check(item['path'], item['sha256'], item['bytes'])
        checkpoints.append(dict(**result, experiment=item['experiment'], variant=item['variant'],
                                seed=item['seed'], epoch=item['epoch'], role='FINAL_REQUIRED'))
    for seed in (2026100201, 2026100202, 2026100203):
        summary_path = f'experiments/C1-04/stage2/seed-{seed}-summary.json'
        check(summary_path)
        item = read(summary_path)['best_checkpoints']['conditioned']
        result = check(item['path'], item['sha256'], item['bytes'])
        checkpoints.append(dict(**result, experiment='E-C01', variant='conditioned', seed=seed,
                                epoch=item['epoch'], role='FINAL_REQUIRED_REDUNDANCY_CONTROL'))
    for item in handoff['selected_checkpoints']:
        if not any(c['sha256'] == item['sha256'] and c['seed'] == item['seed'] and
                   c['variant'] == 'FK_TANH' for c in checkpoints):
            errors.append('selected checkpoint binding')
    if len(checkpoints) != 21:
        errors.append('checkpoint count != 21')
    for path in ('experiments/C1-05/stage2/C1-06-handoff.json',
                 'experiments/C1-05/stage2/results-audit.json',
                 'experiments/C1-05/stage2/NEXT_MODEL_DECISION.md',
                 'experiments/C1-05/stage2/acceptance.json',
                 'experiments/C1-05/stage2/fixed-validation-witness.json',
                 'experiments/C1-02/leakage-audit.json', 'experiments/C1-02/acceptance.json',
                 'src/neurokinematics/benchmark/queries.py',
                 'src/neurokinematics/neural/c104.py', 'src/neurokinematics/neural/c105.py',
                 'src/neurokinematics/neural/physics.py', 'src/neurokinematics/neural/training_fk.py',
                 'experiments/F0-05/benchmark-schema.json'):
        check(path)
    output = dict(task='C1-06', status='PASS' if not errors else 'FAIL', errors=errors,
                  time_utc=datetime.now(timezone.utc).isoformat(), files=list(files.values()),
                  baseline=baseline, checkpoints=checkpoints,
                  query_identity='MANIFEST_HASH_BOUND; per-row join deferred to approved Stage 2',
                  final_test='SEALED_BYTES_ONLY; no row decode or inference',
                  remote_archive='NOT_CONFIRMED',
                  environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform()))
    target = args.output
    if target.exists():
        raise ValueError('preserve prior audit; choose a new attempt path')
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(status=output['status'], files=len(files), baseline_rows=sum(x.get('rows', 0) for x in baseline),
                          checkpoints=len(checkpoints), errors=errors)))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
