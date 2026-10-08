"""Freeze once, then fail closed on C1-05 design/input/evidence drift. No training."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'experiments/C1-05'
REQUIRED = ['config.json', 'STAGE1_REVIEW.md', 'RUN_REPORT.md', 'TEST_MATRIX.md',
            'EXPERIMENT_MATRIX.md', 'COMMANDS.md', 'STORAGE_AND_REPRODUCTION.md',
            'input-hashes.json', 'input-access.json', 'pilot-pair-ids.json',
            'fk-domain-samples.json', 'workspace-start.json', 'environment.json',
            'SOURCE_REQUEST.md']
SOURCE = ['scripts/check_c105_stage1.py', 'scripts/audit_c105_inputs.py',
          'scripts/c105_command.py', 'docs/adr/ADR-013-c105-training-fk-domain.md']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def verify_file(root, relative, expected, size=None):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('missing/outside input: ' + relative)
    if sha(path) != expected or (size is not None and path.stat().st_size != size):
        raise ValueError('SHA/byte drift: ' + relative)


def check_inputs():
    inputs = read('input-hashes.json')['files']
    if len({x['path'] for x in inputs}) != len(inputs):
        raise ValueError('duplicate input')
    for row in inputs:
        verify_file(ROOT, row['path'], row['sha256'], row['bytes'])
    access = read('input-access.json')
    if len(access['checkpoints']) != 6 or access['inventory']['validation']['rows'] != 3600:
        raise ValueError('input inventory drift')
    if access['inventory']['validation']['unlabeled'] != 351:
        raise ValueError('unlabeled validation dropped')
    config = read('config.json')
    if (config['T_C04'] != 'NOT_RUN' or config['training']['seeds'] != [2026100201,2026100202,2026100203]
            or config['search']['preregistered_configurations'] != len(config['matrix'])
            or len(config['matrix']) > config['search']['max_validation_configurations']
            or config['search']['candidates_per_arm'] != 1 or config['search']['adaptive_tuning']):
        raise ValueError('frozen stage/seed/search contract drift')
    return len(inputs)


def freeze():
    target = EVIDENCE / 'SHA256SUMS'
    if target.exists():
        raise ValueError('already frozen; never silently regenerate')
    for name in REQUIRED:
        if not (EVIDENCE / name).is_file():
            raise ValueError('incomplete review: ' + name)
    paths = sorted({p.relative_to(ROOT).as_posix() for p in EVIDENCE.rglob('*') if p.is_file()} | set(SOURCE))
    target.write_text(''.join(f'{sha(ROOT / p)}  {p}\n' for p in paths), encoding='utf-8', newline='\n')
    return len(paths)


def check_frozen():
    rows = (EVIDENCE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    paths = set()
    for line in rows:
        expected, separator, relative = line.partition('  ')
        if not separator or len(expected) != 64 or relative in paths:
            raise ValueError('invalid/duplicate SHA row')
        verify_file(ROOT, relative, expected)
        paths.add(relative)
    needed = {'experiments/C1-05/' + name for name in REQUIRED} | set(SOURCE)
    if not needed <= paths:
        raise ValueError('incomplete SHA closure')
    # Actual command/log consistency is independent of whether a post-freeze log is in the snapshot.
    commands = 0
    for path in (EVIDENCE / 'commands').glob('*/command.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        for key in ('stdout', 'stderr'):
            if sha(path.parent / (key + '.log')) != value[key + '_sha256']:
                raise ValueError('command log drift: ' + str(path))
        commands += 1
    return len(paths), commands


def self_test():
    with tempfile.TemporaryDirectory(prefix='c105-audit-') as folder:
        root = Path(folder)
        path = root / 'fixture'
        path.write_bytes(b'original')
        expected = sha(path)
        verify_file(root, 'fixture', expected, 8)
        path.write_bytes(b'modified')
        caught = 0
        for relative, size in [('fixture', 8), ('missing', None), ('../outside', None)]:
            try:
                verify_file(root, relative, expected, size)
            except ValueError:
                caught += 1
        if caught != 3:
            raise AssertionError('audit mutation survived')
    print(json.dumps(dict(status='PASS', audit_negative_controls=caught, T_C04='NOT_RUN')))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--check', action='store_true')
    group.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    count = check_inputs()
    result = {'status':'PASS', 'input_files':count, 'training':'NOT_RUN', 'T-C04':'NOT_RUN'}
    if args.freeze:
        result['frozen_files'] = freeze()
    else:
        result['frozen_files'], result['command_logs'] = check_frozen()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
