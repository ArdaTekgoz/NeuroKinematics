"""Fail-closed F0-06 evidence checks; no solver or acceptance thresholds live here."""
import hashlib
import json
from pathlib import Path, PurePosixPath

EXPECTED_START = 'e7d211f42496f803688e2a510daca97e102092dc'


def require_same(a, b, label):
    if a != b:
        raise ValueError(f'{label} determinism mismatch')


def verify_handoff(root, inputs):
    required = {'assets/robots/robot_a/robot.urdf', 'assets/robots/robot_a/robot_spec.json',
                'assets/robots/robot_a/manifest.json', 'config/robots/tcp_tool0.json',
                'pixi.lock', 'experiments/F0-04/config.json', 'experiments/F0-04/schema.json',
                'experiments/F0-04/dataset-manifest.json', 'experiments/F0-04/normalization.json',
                'experiments/F0-05/config.json', 'experiments/F0-05/solver-config.json',
                'experiments/F0-05/benchmark-contract.json', 'experiments/F0-05/benchmark-schema.json',
                'experiments/F0-05/query-manifest.json', 'experiments/F0-05/result-manifest.json'}
    if not required <= set(inputs):
        raise ValueError('missing Core handoff input')
    for relative, expected in inputs.items():
        path = Path(root)/relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'invalid Core handoff input: {relative}')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(root, checksum_file):
    root = Path(root).resolve()
    seen = set()
    for line in Path(checksum_file).read_text(encoding='utf-8').splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip('*')
        path = (root / relative).resolve()
        if (PurePosixPath(relative).is_absolute() or not path.is_relative_to(root)
                or relative in seen or path == Path(checksum_file).resolve()):
            raise ValueError('duplicate, unsafe, or self-referential checksum path')
        seen.add(relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'missing or corrupt evidence: {relative}')
    if not seen:
        raise ValueError('empty evidence checksum list')
    return len(seen)


def validate_gate(state):
    required = {'start_commit', 'historical_integrity', 'locked_environment',
                'regression', 'dataset_determinism', 'query_determinism',
                'split_audit', 'normalization', 'fk_limits', 'benchmark_schema',
                'handoff_inputs', 'commands_complete', 'critical_errors',
                'linux_execution', 'linux_claim', 'g0', 'foundations', 'core'}
    if set(state) != required:
        raise ValueError('gate state fields missing or unexpected')
    if state['start_commit'] != EXPECTED_START:
        raise ValueError('wrong start commit')
    for name in ('historical_integrity', 'locked_environment', 'regression',
                 'dataset_determinism', 'query_determinism', 'split_audit',
                 'normalization', 'fk_limits', 'benchmark_schema',
                 'handoff_inputs', 'commands_complete'):
        if state[name] != 'PASS':
            raise ValueError(f'gate requirement not passed: {name}')
    if state['critical_errors'] != []:
        raise ValueError('critical errors remain')
    if state['linux_execution'] == 'NOT_RUN' and state['linux_claim'] != 'NOT_VERIFIED':
        raise ValueError('unexecuted Linux cannot be verified')
    if (state['g0'], state['foundations'], state['core']) != ('PASS / ACCEPTED', 'COMPLETE', 'READY / NOT_STARTED'):
        raise ValueError('inconsistent phase decision')


def validate_index(index):
    for key in ('tasks', 'requirements', 'tests', 'commits', 'evidence', 'platforms',
                'limitations', 'external_evidence', 'gate'):
        if key not in index:
            raise ValueError(f'missing index field: {key}')
    if set(index['tasks']) != {f'F0-0{i}' for i in range(7)}:
        raise ValueError('incomplete task index')
    if set(index['tests']) != {f'T-F{i:02d}' for i in range(10)}:
        raise ValueError('incomplete test index')
    if set(index['requirements']) != {f'REQ-F{i:02d}' for i in range(7)}:
        raise ValueError('incomplete requirements index')
    seen = set()
    for entry in index['evidence']:
        if entry['path'] in seen or len(entry['sha256']) != 64:
            raise ValueError('invalid evidence index')
        int(entry['sha256'], 16)
        seen.add(entry['path'])
    validate_gate(index['gate'])


def audit_dataset(root, manifest, config_path, evidence):
    """Recompute split, train statistics, hard-subset labels and FK from shards."""
    import numpy as np
    from neurokinematics.data.factory import (read_shard, verify_dataset, normalization,
        audit_groups_and_duplicates, validate_boundary, train_singularity_threshold)
    from neurokinematics.kinematics import load_robot, IndependentFK
    from neurokinematics.kinematics.metrics import quaternion_rotation
    from neurokinematics.kinematics.model import ROOT
    config = json.loads(Path(config_path).read_bytes())
    schema = json.loads((ROOT/'experiments/F0-04/schema.json').read_bytes())
    order = [f['name'] for f in schema['fields']]
    verify_dataset(Path(root), Path(root)/'dataset-manifest.json', config_path=Path(config_path))
    subsets = {}
    for name, shards in manifest['shards'].items():
        arrays = [read_shard(Path(root)/s['path'], order) for s in shards]
        subsets[name] = {k: np.concatenate([a[k] for a in arrays]) for k in order}
    combined = {k: np.concatenate([subsets[n][k] for n in ('main', 'boundary', 'singularity')]) for k in order}
    audit_groups_and_duplicates(combined)
    splits = ('train', 'validation', 'test')
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            ma, mb = combined['split'] == a.encode(), combined['split'] == b.encode()
            if set(combined['group_id'][ma]) & set(combined['group_id'][mb]):
                raise ValueError('split group leakage')
            if {tuple(q) for q in combined['q'][ma]} & {tuple(q) for q in combined['q'][mb]}:
                raise ValueError('cross-split exact q duplicate')
    actual = normalization(subsets['main'])
    saved = json.loads((Path(evidence)/'normalization.json').read_bytes())
    if actual != saved:
        raise ValueError('normalization differs from train-only recomputation')
    changed = {k: v.copy() for k, v in subsets['main'].items()}
    for key in ('q', 'position_m'):
        changed[key][changed['split'] != b'train'] += 100
    if normalization(changed) != actual:
        raise ValueError('normalization depends on non-train rows')
    inputs = load_robot()
    validate_boundary(subsets['boundary']['q'], inputs.limits,
                      config['hard_subsets']['boundary']['normalized_joint_limit_distance_lt'])
    threshold = train_singularity_threshold(subsets['main'], config['hard_subsets']['singularity']['threshold_quantile'])
    if np.any(subsets['singularity']['sigma_min'] > threshold):
        raise ValueError('singularity threshold violated')
    fk = IndependentFK(inputs)
    for q, p, quat in zip(combined['q'], combined['position_m'], combined['quaternion_wxyz']):
        pose = fk.forward_kinematics(q)
        if np.linalg.norm(pose[:3, 3]-p) > 1e-9 or np.linalg.norm(pose[:3, :3]-quaternion_rotation(quat), ord='fro') > 1e-9:
            raise ValueError('independent FK mismatch')
    return {'status': 'PASS', 'rows': len(combined['q']), 'cross_split_groups': 0,
            'cross_split_q_duplicates': 0, 'normalization': 'TRAIN_ONLY_RECOMPUTED',
            'singularity_threshold': threshold, 'fk': 'PASS', 'limits_schema': 'PASS'}
