"""Preregistered optimizer/scale/width controls; historical modules immutable."""
from pathlib import Path
import time
import numpy as np
import torch
from . import c106r_diagnostic3 as r

d = r.d
BASE = d.ROOT / 'experiments/C1-06R/diagnostic4'
CONFIG = BASE / 'config.json'


class DiagnosticMLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(13, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 6))

    def forward(self, x):
        return self.layers(x)


def build_model(seed, width, device='cuda'):
    if width not in (256, 512):
        raise ValueError('unregistered width')
    d.setup(seed)
    model = DiagnosticMLP(width).to(device)
    torch.nn.init.zeros_(model.layers[-1].weight)
    torch.nn.init.zeros_(model.layers[-1].bias)
    return model


def q_loss(model, x, y, scale=1.0):
    return ((d.predict(model, x, 'residual') - y)**2).sum(-1).mean() * scale


def train_model(model, x, y, arm, cfg):
    start = time.perf_counter()
    model.train()
    calls, history = 0, []
    if arm['optimizer'] == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), **cfg['adamw'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['steps'], **cfg['schedule'])
    else:
        opt = torch.optim.LBFGS(model.parameters(), **cfg['lbfgs'])

    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True)
        loss = q_loss(model, x, y, arm['loss_scale'])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite or missing gradient')
        calls += 1
        if calls == 1 or calls % 1000 == 0:
            history.append(dict(backward_calls=calls, unscaled_q_loss=float(loss.detach()) / arm['loss_scale']))
            print(f"{arm['name']}: backward {calls}, Q={history[-1]['unscaled_q_loss']:.8g}", flush=True)
        return loss

    if arm['optimizer'] == 'adamw':
        for _ in range(cfg['steps']):
            closure()
            opt.step()
            sch.step()
        iterations = cfg['steps']
    else:
        opt.step(closure)
        state = opt.state[next(iter(model.parameters()))]
        iterations = state['n_iter']
        assert calls == state['func_evals']
    with torch.no_grad():
        terminal = float(q_loss(model, x, y))
    torch.cuda.synchronize()
    return dict(backward_calls=calls, optimizer_iterations=iterations,
                history=history, terminal_q_loss=terminal, training_wall_s=time.perf_counter()-start)


def run():
    d.configure()
    frozen = d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files']
    d.guard_hashes(frozen)
    cfg = d.read_json(CONFIG)
    for name in ('registration.json', 'results.json'):
        if (BASE/name).exists():
            raise FileExistsError('preserve prior attempt: '+name)
    previous = d.read_json(r.BASE/'results.json')
    d.guard_hashes(previous['inputs'])
    source_paths = [Path(__file__), Path(r.__file__), Path(d.__file__),
                    d.ROOT/'src/neurokinematics/neural/c106r_precision.py', CONFIG,
                    d.ROOT/cfg['normalization'], d.ROOT/'docs/adr/ADR-018-c106r-optimization-scale-capacity.md',
                    d.ROOT/'tests/c1_06r/test_diagnostic4.py', d.ROOT/'scripts/audit_c106r_diagnostic4.py']
    inputs = {str(p.relative_to(d.ROOT)):d.sha(p) for p in source_paths}
    d.write_json(BASE/'registration.json', dict(status='REGISTERED_BEFORE_TRAINING', inputs=inputs,
                 inherited_freeze_sha256=d.sha(d.ROOT/'experiments/C1-06R/training-freeze.json'), final_test='NOT_CREATED'))
    train, val = d.load_data(label_fk=True)
    rows = d.matched_rows(train, cfg['n'])[0]['local']
    norm = d.read_json(d.ROOT/cfg['normalization'])
    values, val_values = r.features(rows, 'relative', norm), r.features(val, 'relative', norm)
    x, y = torch.tensor(values, device='cuda'), torch.tensor(rows.target_normalized, device='cuda')
    cells = []
    for arm in cfg['arms']:
        name = arm['name']
        output = d.ROOT/'data/generated/C1-06R/diagnostic4'/name
        output.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        model = build_model(cfg['seed'], arm['width'])
        initial_hash = d.state_hash(model)
        with torch.no_grad():
            assert torch.equal(d.predict(model, x, 'residual'), x[:, -6:])
        fit = train_model(model, x, y, arm, cfg)
        measured = dict(train=r.evaluate(model, 'residual', rows, values),
                        validation=r.evaluate(model, 'residual', val, val_values))
        weight = output/'last.pt'
        contract = dict(arm=arm, seed=cfg['seed'], n=cfg['n'], features=cfg['features'], head=cfg['head'], inputs=inputs,
                        torch=str(torch.__version__), cuda=str(torch.version.cuda), pair_ids=rows.pair_id.tolist())
        torch.save(dict(contract=contract, model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}), weight)
        payload = torch.load(weight, weights_only=True)
        assert payload['contract'] == contract and type(payload['contract']['torch']) is str
        restored = build_model(cfg['seed'], arm['width'])
        restored.load_state_dict(payload['model_state_dict'])
        for split, split_rows, split_values in (('train', rows, values), ('validation', val, val_values)):
            assert r.evaluate(restored, 'residual', split_rows, split_values) == measured[split]
        d.write_json(output/'predictions.json', measured)
        result = dict(**arm, n=cfg['n'], seed=cfg['seed'], initial_hash=initial_hash,
                      parameters=sum(p.numel() for p in model.parameters()), **fit,
                      train=r.summary(measured['train'], rows), validation=r.summary(measured['validation'], val),
                      checkpoint=dict(path=str(weight.relative_to(d.ROOT)), sha256=d.sha(weight)),
                      raw=dict(path=str((output/'predictions.json').relative_to(d.ROOT)), sha256=d.sha(output/'predictions.json')),
                      reload='EXACT_MATCH_TRAIN_AND_VALIDATION', wall_s=time.perf_counter()-start)
        d.write_json(BASE/(name+'.json'), result)
        cells.append(result)
        print(f"{name}: train A {result['train']['profile_a']}/{cfg['n']}; validation A {result['validation']['profile_a']}/3600", flush=True)
    d.guard_hashes(inputs)
    d.guard_hashes(frozen)
    d.write_json(BASE/'results.json', dict(status='COMPLETE_DIAGNOSIS', cells=cells, inputs=inputs,
                 single_seed=True, long_training='NOT_RUN', final_test='NOT_CREATED'))


if __name__ == '__main__':
    run()
