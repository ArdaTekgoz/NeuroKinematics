"""Verify full paired matrix and exact replay of the four previous raw controls."""
from itertools import product
from pathlib import Path
import numpy as np
import torch
from neurokinematics.neural import c106r_diagnostic3 as r


def main():
    out = r.BASE/'audit.json'
    if out.exists():
        raise FileExistsError(out)
    cfg = r.d.read_json(r.CONFIG)
    results = r.d.read_json(r.BASE/'results.json')
    registration = r.d.read_json(r.BASE/'registration.json')
    assert results['inputs'] == registration['inputs']
    r.d.guard_hashes(results['inputs'])
    r.d.guard_hashes(r.d.read_json(r.d.ROOT/'experiments/C1-06R/training-freeze.json')['files'])
    expected = set(product(cfg['sizes'],cfg['mixtures'],cfg['features'],cfg['heads']))
    cells = results['cells']
    actual = [(c['n'],c['mixture'],c['features'],c['head']) for c in cells]
    assert len(actual) == len(expected) and set(actual) == expected
    assert len(set(c['initial_hash'] for c in cells)) == 1
    assert results['total_updates'] == sum(c['steps'] for c in cells) == 80000
    old = r.d.read_json(r.d.BASE/'results.json')['cells']
    raw_replay = []
    identities = {}
    for cell in cells:
        assert cell['steps'] == cfg['steps'] and cell['seed'] == cfg['seed']
        assert [h['step'] for h in cell['history']] == list(range(0,cfg['steps']+1,cfg['log_every']))
        for key in ('checkpoint','raw'):
            assert r.d.sha(r.d.ROOT/cell[key]['path']) == cell[key]['sha256']
        weight = torch.load(r.d.ROOT/cell['checkpoint']['path'],weights_only=True)
        assert weight['contract']['inputs'] == results['inputs']
        ids = weight['contract']['pair_ids']
        assert len(ids) == len(set(ids)) == cell['n']
        pairkey = (cell['n'],cell['mixture'])
        if pairkey in identities:
            assert identities[pairkey] == ids
        identities[pairkey] = ids
        raw = r.d.read_json(r.d.ROOT/cell['raw']['path'])
        for split,n in (('train',cell['n']),('validation',3600)):
            metric = raw[split]
            assert metric['n'] == cell[split]['n'] == len(metric['rows']) == n
            assert len({x['pair_id'] for x in metric['rows']}) == n
            assert sum(x['profile_a'] for x in metric['rows']) == cell[split]['profile_a']
            assert sum(x['profile_b'] for x in metric['rows']) == cell[split]['profile_b']
            for x in metric['rows']:
                assert x['profile_a'] == bool(x['valid'] and x['position_m'] <= .002 and x['orientation_deg'] <= 1)
                assert x['profile_b'] == bool(x['valid'] and x['position_m'] <= .001 and x['orientation_deg'] <= .5)
            if split == 'train':
                assert [x['pair_id'] for x in metric['rows']] == ids
        if cell['n'] == 512 and cell['features'] == 'raw':
            previous = next(x for x in old if x['n']==512 and x['mixture']==cell['mixture'] and x['head']==cell['head'])
            prior = torch.load(r.d.ROOT/previous['checkpoint']['path'],weights_only=True)
            assert r.d.sha(r.d.ROOT/previous['checkpoint']['path']) == previous['checkpoint']['sha256']
            assert set(prior['model_state_dict']) == set(weight['model_state_dict'])
            assert all(torch.equal(prior['model_state_dict'][k],v) for k,v in weight['model_state_dict'].items())
            raw_replay.append(dict(name=cell['name'],state_tensors='EXACT_MATCH_DIAGNOSTIC2'))
    comparisons = []
    for n,mix,head in product(cfg['sizes'],cfg['mixtures'],cfg['heads']):
        a,b = [next(c for c in cells if (c['n'],c['mixture'],c['features'],c['head'])==(n,mix,mode,head)) for mode in cfg['features']]
        comparisons.append(dict(n=n,mixture=mix,head=head,raw_train_a=a['train']['profile_a'],relative_train_a=b['train']['profile_a'],
           raw_validation_a=a['validation']['profile_a'],relative_validation_a=b['validation']['profile_a'],
           raw_local=a['validation']['strata']['local'],relative_local=b['validation']['strata']['local'],
           raw_wide=a['validation']['strata']['wide'],relative_wide=b['validation']['strata']['wide']))
    r.d.write_json(out,dict(status='PASS',cells=len(cells),updates=results['total_updates'],raw_controls=raw_replay,
                 comparisons=comparisons,inputs=results['inputs'],code_sha256=r.d.sha(Path(__file__)),
                 max_main_validation_a=max(c['validation']['strata']['main']['profile_a'] for c in cells),
                 required_product_gate='three seeds each main A >=95%; single-seed diagnosis cannot pass',
                 final_test='NOT_CREATED',old_final_raw='NOT_READ'))
    print(f'PASS: {len(cells)} cells, 80000 updates, four exact raw weight replays',flush=True)


if __name__ == '__main__':
    main()
