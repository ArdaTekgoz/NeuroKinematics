"""Measure corrected decoder on oracle and old models without rewriting them."""
from pathlib import Path
import numpy as np
import torch
from neurokinematics.neural import c106r_diagnostic2 as d, c106r_training as t
from neurokinematics.neural.c106r_precision import decode_joints


def measure(model, rows, predict):
    q = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows.pair_id), 1024):
            x = torch.tensor(rows.conditioned[start:start+1024], device='cuda')
            q.append(decode_joints(predict(model, x)).cpu().numpy())
    return d.geometric_metrics(np.concatenate(q), rows)


def main():
    d.configure()
    output = d.BASE / 'project-review/precision-fix.json'
    if output.exists():
        raise FileExistsError(output)
    d.guard_hashes(d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'])
    train, val = d.load_data(label_fk=True)
    oracle = {}
    for rows in (train, val):
        labeled = rows.take(np.flatnonzero(rows.label_present))
        q = decode_joints(torch.tensor(labeled.target_normalized, device='cuda')).cpu().numpy()
        oracle[rows.split] = d.geometric_metrics(q, labeled)
        assert oracle[rows.split]['profile_a'] == len(labeled.pair_id)
        assert oracle[rows.split]['out_of_limits'] == 0
    old = d.read_json(d.ROOT/'data/generated/C1-06R/round1/campaign-complete.json')
    runs = []
    for run in old['runs']:
        path = Path(run['best_checkpoint']['path'])
        assert d.sha(path) == run['best_checkpoint']['sha256']
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            payload = torch.load(path, weights_only=True)
        model = d.MLP('conditioned').cuda()
        model.load_state_dict(payload['model_state_dict'])
        actual = measure(model, val, lambda m,x:t.normalized(m(x), run['arm']))
        runs.append(dict(seed=run['seed'],arm=run['arm'],source_sha256=d.sha(path),
                         prior=run['best']['validation'],corrected=actual))
    cells = []
    for cell in d.read_json(d.BASE/'results.json')['cells']:
        path = d.ROOT/cell['checkpoint']['path']
        assert d.sha(path) == cell['checkpoint']['sha256']
        model = d.build_model(cell['seed'])
        model.load_state_dict(torch.load(path,weights_only=True)['model_state_dict'])
        fn = lambda m,x:d.predict(m,x,cell['head'])
        selected = d.matched_rows(train,cell['n'])[0][cell['mixture']]
        cells.append(dict(name=cell['name'],prior_train_a=cell['train']['profile_a'],
                          train=measure(model,selected,fn),validation=measure(model,val,fn)))
    d.write_json(output,dict(status='PASS',oracle=oracle,round1=runs,diagnostic2=cells,
                 historical_results='UNCHANGED',code_sha256=d.sha(Path(__file__)),
                 decoder_sha256=d.sha(d.ROOT/'src/neurokinematics/neural/c106r_precision.py')))
    print('PASS: corrected endpoint decoder; old model impact recorded independently',flush=True)


if __name__ == '__main__':
    main()
