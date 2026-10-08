"""Read-only C1-05 input audit. Hash sealed bytes; decode train/validation only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from neurokinematics.neural.c104 import load_data, load_checkpoint
from neurokinematics.kinematics.model import load_robot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/C1-05'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    path = OUT / name
    if path.exists():
        raise ValueError(f'Preserve previous audit: {path}')
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def main():
    files = {}

    def check(path, expected=None, size=None):
        p = ROOT / path
        actual = digest(p)
        if expected is not None and actual != expected:
            raise ValueError(f'SHA drift: {path}: {actual} != {expected}')
        if size is not None and p.stat().st_size != size:
            raise ValueError(f'Byte count drift: {path}')
        files[path] = dict(path=path, sha256=actual, bytes=p.stat().st_size,
                           storage='LOCAL_ONLY' if path.startswith('data/generated/') else 'GIT', accessible=True)
        return files[path]

    # Established robot, C1-02 shard, normalization and accepted FK identities.
    for row in read('experiments/C1-04/input-hashes.json')['files']:
        check(row['path'], row['sha256'], row['bytes'])
    # Verify reproducible source data/solver inputs; do not run generation or test evaluation.
    reproduction = []
    for path, expected in read('experiments/C1-02/input-hashes.json')['files'].items():
        if path == '.gitattributes':
            continue  # historical policy gained C1-03/04 entries; not a generation input
        reproduction.append(check(path, expected))
    for path in ('experiments/C1-02/COMMANDS.md', 'scripts/run_c102_full.py',
                 'scripts/run_c102_stage2.ps1', 'scripts/compare_c102_reproduction.py',
                 'src/neurokinematics/neural/c104.py', 'src/neurokinematics/kinematics/chain.py',
                 'experiments/C1-04/config.json', 'experiments/C1-04/checkpoint-schema.json',
                 'experiments/C1-04/input-hashes.json', 'experiments/C1-04/SHA256SUMS',
                 'experiments/C1-04/stage2/evidence-manifest.json',
                 'experiments/C1-03/config.json', 'experiments/C1-03/samples.jsonl'):
        check(path)
    for row in read('experiments/C1-04/stage2/evidence-manifest.json')['files']:
        # Preserve exact historical evidence identity, including six per-row files.
        p = ROOT / row['path']
        canonical = hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if canonical != row['canonical_lf_sha256']:
            raise ValueError('C1-04 evidence drift: ' + row['path'])
        check(row['path'])

    acceptance = read('experiments/C1-04/stage2/acceptance.json')
    if (acceptance['T-C03'] != 'PASS' or acceptance['direct_ik_decision'] != 'NO_GO'
            or acceptance['E-C01'] != 'THREE_PAIRED_SEEDS_COMPLETE'):
        raise ValueError('historical acceptance drift')
    if read('experiments/C1-04/stage2/pilot-summary.json')['t_c03_status'] != 'PASS':
        raise ValueError('historical pilot drift')
    train, val = load_data(label_fk=True)
    inventory = {}
    for rows in (train, val):
        order = ('\n'.join(rows.pair_id.tolist()) + '\n').encode()
        labeled = ('\n'.join(rows.pair_id[rows.label_present].tolist()) + '\n').encode()
        inventory[rows.split] = dict(rows=len(rows.pair_id), labeled=int(rows.label_present.sum()),
                                    unlabeled=int((~rows.label_present).sum()),
                                    modes=dict(Counter(rows.mode.tolist())),
                                    labeled_modes=dict(Counter(rows.mode[rows.label_present].tolist())),
                                    families=dict(Counter(rows.family.tolist())),
                                    ordered_pair_ids_sha256=hashlib.sha256(order).hexdigest(),
                                    labeled_ordered_pair_ids_sha256=hashlib.sha256(labeled).hexdigest())
    torch.set_num_threads(1)
    checkpoints, outcomes = [], []
    for outcome in acceptance['outcomes']:
        seed, variant = outcome['seed'], outcome['variant']
        summary = read(f'experiments/C1-04/stage2/seed-{seed}-summary.json')
        if summary['epochs'] != 200 or summary['optimizer_steps_per_model'] != 3000:
            raise ValueError('C1-04 training budget drift')
        old = summary['best_checkpoints'][variant]
        relative = f'data/generated/C1-04/v1/seed-{seed}/{variant}-best.pt'
        if old['sha256'] != outcome['checkpoint_sha256']:
            raise ValueError('checkpoint acceptance binding drift')
        item = dict(check(relative, old['sha256'], old['bytes']), seed=seed, variant=variant,
                    epoch=old['epoch'], historical_absolute_path=old['path'])
        model, metadata = load_checkpoint(ROOT / relative, expected_variant=variant)
        # Reproduce all existing validation q outputs; no new model/metric tuning.
        lower, upper = np.asarray(load_robot().limits, dtype=np.float64).T
        with torch.no_grad():
            features = val.feature(variant)
            predictions = [model(torch.from_numpy(features[i:i + 1024])).numpy().astype(np.float64)
                           for i in range(0, len(features), 1024)]
        q = lower + np.concatenate(predictions) * (upper - lower)
        raw_path = f'experiments/C1-04/stage2/seed-{seed}-{variant}-validation.jsonl'
        raw = [json.loads(line) for line in (ROOT / raw_path).read_text(encoding='utf-8').splitlines()]
        stat = read(raw_path.replace('.jsonl', '.summary.json'))
        check(raw_path, stat['rows_sha256'], stat['rows_bytes'])
        if [r['pair_id'] for r in raw] != val.pair_id.tolist():
            raise ValueError('validation inventory drift')
        if [r['label_present'] for r in raw] != val.label_present.tolist():
            raise ValueError('validation label mask drift')
        previous_q = np.asarray([r['q_raw_rad'] for r in raw])
        diff = float(np.max(np.abs(q - previous_q)))
        if diff != 0:
            raise ValueError(f'historical inference drift: {seed}/{variant}: {diff}')
        successes = sum(r['finite'] and r['in_limits'] and r['position_error_m'] <= .002
                        and r['orientation_error_deg'] <= 1 for r in raw)
        valid = sum(r['in_limits'] for r in raw)
        if successes != outcome['profile_a_success'] or valid != outcome['valid_raw']:
            raise ValueError('historical outcome drift')
        item.update(load_status='PASS', validation_predictions=len(raw), max_q_abs_rad=diff)
        checkpoints.append(item)
        positions = [r['position_error_m'] for r in raw if r['in_limits']]
        angles = [r['orientation_error_deg'] for r in raw if r['in_limits']]
        outcomes.append(dict(seed=seed, variant=variant, total=len(raw), labeled=3249, unlabeled=351,
                             valid_raw=valid, out_of_limits=outcome['out_of_limits'], profile_a=successes,
                             valid_only_position_median_m=float(np.median(positions)),
                             valid_only_position_p95_m=float(np.percentile(positions, 95)),
                             valid_only_orientation_median_deg=float(np.median(angles)),
                             valid_only_orientation_p95_deg=float(np.percentile(angles, 95))))
    # Freeze pilot by identities before any C1-05 result: 32 labeled roots per family/mode.
    pilot = {}
    for rows in (train, val):
        ids = []
        count = 32 if rows.split == 'train' else 16
        for family in ('main', 'boundary', 'singularity'):
            for mode in ('local', 'wide'):
                mask = (rows.family == family) & (rows.mode == mode) & rows.label_present
                ids.extend(rows.pair_id[np.flatnonzero(mask)[:count]].tolist())
        pilot[rows.split] = sorted(ids)
    check('pixi.toml')
    write('input-hashes.json', dict(task='C1-05', files=sorted(files.values(), key=lambda r: r['path'])))
    write('input-access.json', dict(status='PASS', inventory=inventory, checkpoints=checkpoints,
          baseline_outcomes=outcomes, reproduction_input_count=len(reproduction),
          reproduction='SOURCE_BYTES_VERIFIED; generation NOT_RUN in C1-05',
          sealed_data='test/benchmark bytes hashed only; no decoding/evaluation', remote_archive='NOT_CONFIRMED',
          historical_inference='21600 predictions exactly reproduced', new_training='NOT_RUN'))
    write('pilot-pair-ids.json', pilot)
    write('environment.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
          torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
          git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    print(json.dumps(dict(status='PASS', input_files=len(files), checkpoints=len(checkpoints),
                          historical_predictions=21600, inventory=inventory, pilot_counts={k:len(v) for k,v in pilot.items()})))


if __name__ == '__main__':
    main()
