"""Audit four arms, exact historical reference, raw denominators and thresholds."""
from pathlib import Path
import torch
from neurokinematics.neural import c106r_diagnostic4 as e


def main():
    d = e.d
    out = e.BASE/'audit.json'
    if out.exists():
        raise FileExistsError(out)
    cfg = d.read_json(e.CONFIG)
    results = d.read_json(e.BASE/'results.json')
    registration = d.read_json(e.BASE/'registration.json')
    assert results['inputs'] == registration['inputs']
    d.guard_hashes(results['inputs'])
    frozen = d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files']
    d.guard_hashes(frozen)
    old = d.read_json(e.r.BASE/'results.json')
    d.guard_hashes(old['inputs'])
    previous = next(c for c in old['cells'] if c['name'] == 'n2048-local-relative-residual')
    assert d.sha(d.ROOT/previous['checkpoint']['path']) == previous['checkpoint']['sha256']
    prior = torch.load(d.ROOT/previous['checkpoint']['path'], weights_only=True)
    cells = results['cells']
    assert [c['name'] for c in cells] == [a['name'] for a in cfg['arms']]
    assert len(set(c['initial_hash'] for c in cells if c['width'] == 256)) == 1
    calls = {}
    for cell, arm in zip(cells, cfg['arms']):
        assert all(cell[k] == v for k,v in arm.items())
        assert cell['n'] == 2048 and cell['seed'] == cfg['seed']
        assert cell['reload'] == 'EXACT_MATCH_TRAIN_AND_VALIDATION'
        if arm['optimizer'] == 'adamw':
            assert cell['backward_calls'] == cell['optimizer_iterations'] == cfg['steps']
        else:
            assert 0 < cell['optimizer_iterations'] <= cfg['lbfgs']['max_iter']
            assert cell['backward_calls'] > 0
        calls[cell['name']] = {k:cell[k] for k in ('optimizer_iterations','backward_calls','training_wall_s')}
        for key in ('checkpoint', 'raw'):
            assert d.sha(d.ROOT/cell[key]['path']) == cell[key]['sha256']
        payload = torch.load(d.ROOT/cell['checkpoint']['path'], weights_only=True)
        assert payload['contract']['inputs'] == results['inputs']
        ids = payload['contract']['pair_ids']
        assert ids == prior['contract']['pair_ids'] and len(set(ids)) == 2048
        raw = d.read_json(d.ROOT/cell['raw']['path'])
        for split, n in (('train',2048), ('validation',3600)):
            m = raw[split]
            assert m['n'] == cell[split]['n'] == len(m['rows']) == n
            assert len({x['pair_id'] for x in m['rows']}) == n
            for profile, pos, ang in (('profile_a',.002,1), ('profile_b',.001,.5)):
                assert sum(x[profile] for x in m['rows']) == cell[split][profile]
                for x in m['rows']:
                    assert x[profile] == bool(x['valid'] and x['position_m'] <= pos and x['orientation_deg'] <= ang)
            if split == 'train':
                assert [x['pair_id'] for x in m['rows']] == ids
            else:
                for group in ('local','wide'):
                    assert cell[split]['strata'][group]['n'] == 1800
        if cell['name'] == 'reference':
            assert set(payload['model_state_dict']) == set(prior['model_state_dict'])
            assert all(torch.equal(prior['model_state_dict'][k], v) for k,v in payload['model_state_dict'].items())
            assert cell['train'] == previous['train'] and cell['validation'] == previous['validation']
    d.write_json(out, dict(status='PASS', cells=4, inherited_frozen_files=len(frozen),
                 reference='EXACT_TENSORS_AND_METRICS_DIAGNOSTIC3', paired_train_ids=2048,
                 safe_reload_rows=4*(2048+3600), measured_budgets=calls,
                 inputs=results['inputs'], code_sha256=d.sha(Path(__file__)),
                 product_gate='NOT_MET' if max(c['validation']['strata']['main']['profile_a'] for c in cells) < 2850 else 'SINGLE_SEED_INSUFFICIENT',
                 final_test='NOT_CREATED', old_final_raw='NOT_READ'))
    print('PASS: four controls, exact reference replay, full denominators, hashes and unchanged A/B thresholds')


if __name__ == '__main__':
    main()
