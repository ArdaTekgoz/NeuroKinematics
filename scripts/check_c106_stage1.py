"""Freeze and verify Stage 1, including sealed byte hashes; no final decoding."""
import argparse
import json
from pathlib import Path
from audit_c106_inputs import fingerprint

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/C1-06'
REQUIRED = ['SOURCE_REQUEST.md', 'config.json', 'PROTOCOL.md', 'COMPARISON_MATRIX.md',
            'COMMANDS.md', 'STORAGE_AND_REPRODUCTION.md', 'RUN_REPORT.md',
            'input-hashes.json', 'test-seal.json', 'stage1-decision.json',
            'synthetic-tests.xml', 'synthetic-tests-final.xml', 'synthetic-smoke.jsonl', 'start.json']
SOURCES = ['src/neurokinematics/neural/c106.py', 'tests/c1_06/test_protocol.py',
           'scripts/audit_c106_inputs.py', 'scripts/check_c106_stage1.py',
           'scripts/c106_command.py', 'scripts/smoke_c106.py']


def verify_row(row):
    path = (ROOT / row['path']).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('outside repository: ' + row['path'])
    actual = fingerprint(path, 'rows' in row)
    if any(actual[k] != row[k] for k in ('sha256', 'bytes', 'rows') if k in row):
        raise ValueError('input drift: ' + row['path'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    if args.freeze == args.check:
        p.error('choose --freeze or --check')
    audit = json.loads((BASE / 'input-hashes.json').read_bytes())
    if audit['status'] != 'PASS':
        raise ValueError('input audit did not pass')
    for row in audit['files']:
        verify_row(row)
    needed = {'experiments/C1-06/' + name for name in REQUIRED} | set(SOURCES)
    for rel in needed:
        if not (ROOT / rel).is_file():
            raise ValueError('missing protocol file: ' + rel)
    manifest = BASE / 'SHA256SUMS'
    if args.freeze:
        # Never include the command currently running: its logs are not yet written.
        paths = needed | {p.relative_to(ROOT).as_posix() for p in BASE.rglob('*') if p.is_file()}
        with manifest.open('x', encoding='utf-8', newline='\n') as stream:
            for rel in sorted(paths):
                stream.write(f"{fingerprint(ROOT / rel)['sha256']}  {rel}\n")
    seen = set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        if rel in seen:
            raise ValueError('duplicate frozen path')
        seen.add(rel)
        verify_row(dict(path=rel, sha256=digest))
    if not needed <= seen:
        raise ValueError('incomplete protocol closure')
    commands = 0
    for file in (BASE / 'commands').glob('*/command.json'):
        record = json.loads(file.read_bytes())
        for key in ('stdout', 'stderr'):
            if fingerprint(file.parent / (key + '.log'))['sha256'] != record[key + '_sha256']:
                raise ValueError('command log drift')
        commands += 1
    print(json.dumps(dict(status='PASS', inputs=len(audit['files']), frozen_files=len(seen),
                          command_logs=commands, final_test='SEALED_NOT_RUN', T_C05='NOT_RUN')))


if __name__ == '__main__':
    main()
