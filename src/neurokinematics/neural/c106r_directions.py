"""C1-02R/v1: paired repeated-example versus local-direction coverage."""
from dataclasses import replace
from pathlib import Path
import hashlib
import time

import numpy as np
import torch

from . import c106r_diagnostic5 as s
from .c104 import validate_rows

d, r, e = s.d, s.r, s.e
BASE = d.ROOT / 'experiments/C1-06R/diagnostic6'
RAW = d.ROOT / 'data/generated/C1-06R/diagnostic6'
CONFIG = BASE / 'config.json'


def direction_seed(seed, root, role, index):
    value = f'C1-02R/v1|{seed}|{root}|{role}|{index}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(value).digest()[:16], 'little')


def perturb(q, bounds, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    for attempt in range(1, 10001):
        value = q + rng.uniform(-.1, .1, 6)
        if np.all(value >= bounds[:, 0]) and np.all(value <= bounds[:, 1]) and not np.array_equal(value, q):
            return value, attempt
    raise ValueError('local rejection budget exhausted')


def generate(roots, count, role, seed):
    if roots.split != 'train' or role not in ('train', 'probe') or count < 1:
        raise ValueError('train roots and a known role required')
    if not np.all((roots.family == 'main') & (roots.mode == 'local') & roots.label_present):
        raise ValueError('main/local teachers required')
    if len(set(roots.source_sample_id.tolist())) != len(roots.pair_id):
        raise ValueError('one source row per root required')
    validate_rows(roots)
    rows = roots.take(np.repeat(np.arange(len(roots.pair_id)), count))
    bounds = np.asarray(r.load_robot().limits)
    currents, ids, provenance = [], [], []
    for i, root in enumerate(roots.source_sample_id):
        for j in range(count):
            original = role == 'train' and j == 0
            value_seed = direction_seed(seed, str(root), role, j)
            q, attempts = (roots.q_current[i].copy(), 0) if original else perturb(roots.q_target[i], bounds, value_seed)
            identity = f'C1-02R-v1:{root}:{role}:{j}'
            currents.append(q); ids.append(identity)
            provenance.append(dict(pair_id=identity, source_pair_id=str(roots.pair_id[i]),
                root=str(root), role=role, direction=j, seed=None if original else str(value_seed),
                attempts=attempts, original=original))
    current = np.asarray(currents)
    conditioned = np.concatenate((rows.pose_only, (current-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])), axis=1).astype(np.float32)
    result = replace(rows, pair_id=np.asarray(ids), q_current=current, conditioned=conditioned)
    validate_rows(result)
    return result, provenance


def check_separation(roots, directions, probe, validation):
    for field in ('source_sample_id', 'group_id'):
        chosen = set(getattr(roots, field).tolist())
        if chosen & set(getattr(validation, field).tolist()):
            raise ValueError('validation root/group leakage')
        if set(getattr(directions, field).tolist()) != chosen or set(getattr(probe, field).tolist()) != chosen:
            raise ValueError('derived root set drift')
    def keys(rows):
        return {(str(root), q.tobytes()) for root, q in zip(rows.source_sample_id, rows.q_current)}
    a, b = keys(directions), keys(probe)
    if len(a) != len(directions.pair_id) or len(b) != len(probe.pair_id) or a & b:
        raise ValueError('duplicate or overlapping directions')


def artifact(path):
    return dict(path=str(path.relative_to(d.ROOT)), sha256=d.sha(path))


def run():
    d.configure()
    if (BASE/'registration.json').exists() or RAW.exists():
        raise FileExistsError('preserve prior attempt')
    cfg = d.read_json(CONFIG)
    frozen = d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files']
    d.guard_hashes(frozen)
    # Guard the registered feature/model/geometry helpers before importing their evidence.
    for name in ('diagnostic3', 'diagnostic4', 'diagnostic5'):
        d.guard_hashes(d.read_json(d.ROOT/f'experiments/C1-06R/{name}/registration.json')['inputs'])
    paths = [Path(__file__), CONFIG, d.ROOT/'tests/c1_06r/test_directions.py',
             d.ROOT/'docs/adr/ADR-021-c106r-local-direction-coverage.md',
             d.ROOT/cfg['normalization'], d.ROOT/cfg['geometry_contract']]
    inputs = {str(p.relative_to(d.ROOT)): d.sha(p) for p in paths}
    d.write_json(BASE/'registration.json', dict(status='REGISTERED_BEFORE_GENERATION_AND_TRAINING', inputs=inputs))
    RAW.mkdir(parents=True)
    train, val = d.load_data(label_fk=True)
    roots = d.matched_rows(train, max(cfg['sizes']))[0]['local']
    directions, meta = generate(roots, cfg['directions'], 'train', cfg['data_seed'])
    probe, probe_meta = generate(roots, cfg['probe_directions'], 'probe', cfg['data_seed'])
    check_separation(roots, directions, probe, val)
    checks = {}
    for name, rows, count, role in (('directions', directions, cfg['directions'], 'train'), ('probe', probe, cfg['probe_directions'], 'probe')):
        replay, _ = generate(roots, count, role, cfg['data_seed'])
        assert np.array_equal(rows.q_current, replay.q_current)
        _, _, _, checked = s.preflight(rows, d.read_json(d.ROOT/cfg['geometry_contract']), r.load_robot())
        oracle = d.geometric_metrics(rows.q_target, rows, details=False)
        assert oracle['profile_a'] == oracle['profile_b'] == len(rows.pair_id)
        checks[name] = dict(**checked, teacher_oracle=oracle, generation_replay='EXACT_MATCH')
        np.savez_compressed(RAW/(name+'.npz'), **{k:getattr(rows,k) for k in rows.__dataclass_fields__})
    assert np.array_equal(directions.q_current[::cfg['directions']], roots.q_current)
    d.write_json(RAW/'provenance.json', dict(data_version=cfg['revision'], directions=meta, probe=probe_meta))
    d.write_json(BASE/'preflight.json', dict(status='PASS',checks=checks,root_group_separation='PASS',
        unique_roots=len(roots.pair_id), artifacts=[artifact(RAW/p) for p in ('directions.npz','probe.npz','provenance.json')]))
    print('PASS: generation replay, root separation, independent geometry, teacher oracle', flush=True)
    norm = d.read_json(d.ROOT/cfg['normalization'])
    vx = r.features(val, 'relative', norm)
    cells, initial = [], None
    for n in cfg['sizes']:
        original = roots.take(np.arange(n))
        multi = directions.take(np.arange(n*cfg['directions']))
        holdout = probe.take(np.arange(n*cfg['probe_directions']))
        common = dict(original=original, same_root_probe=holdout, validation=val)
        common_x = {k: r.features(rows, 'relative', norm) if k != 'validation' else vx for k, rows in common.items()}
        for arm in cfg['arms']:
            tick = time.perf_counter(); name = f'n{n}-{arm}'
            out = RAW/name; out.mkdir()
            selected = multi if arm == 'DIRECTIONS' else original.take(np.repeat(np.arange(n), cfg['directions']))
            # Occurrence IDs preserve full exposure accounting without claiming uniqueness of inputs.
            if arm == 'REPEAT':
                selected = replace(selected, pair_id=np.asarray([f'{p}:repeat:{i%cfg["directions"]}' for i,p in enumerate(selected.pair_id)]))
            datasets = dict(train=selected, **common)
            values = dict(train=r.features(selected, 'relative', norm), **common_x)
            model = e.build_model(cfg['seed'], cfg['width'])
            digest = d.state_hash(model); initial = digest if initial is None else initial
            assert digest == initial
            x = torch.tensor(values['train'], device='cuda'); y = torch.tensor(selected.target_normalized, device='cuda')
            opt = torch.optim.AdamW(model.parameters(), **cfg['optimizer'])
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['steps'], **cfg['schedule'])
            history = []
            for step in range(1, cfg['steps']+1):
                model.train(); opt.zero_grad(set_to_none=True)
                loss = ((d.predict(model,x,'residual')-y)**2).sum(-1).mean()
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                opt.step(); sch.step()
                if step%1000 == 0:
                    history.append(dict(step=step, pre_update_q_loss=float(loss.detach())))
                    print(f'{name}: {step}/{cfg["steps"]}', flush=True)
            measured = {k:r.evaluate(model,'residual',rows,values[k]) for k,rows in datasets.items()}
            contract = dict(inputs=inputs, name=name, torch=str(torch.__version__), cuda=str(torch.version.cuda),
                            seed=cfg['seed'], train_pair_ids=selected.pair_id.tolist(), checkpoint='terminal_only')
            weight = out/'last.pt'
            torch.save(dict(contract=contract,model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),weight)
            saved = torch.load(weight,weights_only=True); assert saved['contract'] == contract
            restored = e.build_model(cfg['seed'],cfg['width']); restored.load_state_dict(saved['model_state_dict'])
            for k,rows in datasets.items(): assert r.evaluate(restored,'residual',rows,values[k]) == measured[k]
            d.write_json(out/'predictions.json',measured)
            cell = dict(name=name,n_roots=n,arm=arm,unique_train_inputs=n if arm=='REPEAT' else n*cfg['directions'],
                batch_size=len(selected.pair_id),steps=cfg['steps'],exposures=len(selected.pair_id)*cfg['steps'],
                initial_hash=initial,history=history,metrics={k:r.summary(measured[k],rows) for k,rows in datasets.items()},
                checkpoint=artifact(weight),raw=artifact(out/'predictions.json'),reload='EXACT_MATCH_ALL_SETS',wall_s=time.perf_counter()-tick)
            d.write_json(BASE/(name+'.json'),cell);cells.append(cell)
            print(f'{name}: original A{measured["original"]["profile_a"]}/{n}; same-root A{measured["same_root_probe"]["profile_a"]}/{len(holdout.pair_id)}; validation A{measured["validation"]["profile_a"]}/3600',flush=True)
    d.guard_hashes(inputs);d.guard_hashes(frozen)
    d.write_json(BASE/'results.json',dict(status='COMPLETE_DIAGNOSIS',inputs=inputs,cells=cells,
        total_updates=len(cells)*cfg['steps'],single_seed=True,old_final_raw='NOT_READ',final_test='NOT_CREATED'))


if __name__ == '__main__':
    run()
