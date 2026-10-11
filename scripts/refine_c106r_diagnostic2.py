"""Registered same-checkpoint/objective precision diagnosis for n512 cells."""
from pathlib import Path
import time
import torch
from neurokinematics.neural import c106r_diagnostic2 as d


def main():
    d.configure()
    cfgpath = d.BASE / "refinement-config.json"
    cfg = d.read_json(cfgpath)
    source = d.read_json(d.ROOT / cfg["source"])
    out = d.BASE / "refinement"
    out.mkdir(exist_ok=False)
    d.write_json(out / "registration.json", dict(config_sha256=d.sha(cfgpath),
                 source_sha256=d.sha(d.ROOT / cfg["source"]), code_sha256=d.sha(Path(__file__))))
    train, val = d.load_data(label_fk=True)
    sets, common = d.matched_rows(train, cfg["size"])
    results = []
    for cell in source["cells"]:
        if cell["n"] != cfg["size"]:
            continue
        rows, head = sets[cell["mixture"]], cell["head"]
        path = d.ROOT / cell["checkpoint"]["path"]
        assert d.sha(path) == cell["checkpoint"]["sha256"]
        payload = torch.load(path, weights_only=True)
        assert payload["contract"] == cell["contract"]
        assert payload["contract"]["pair_ids"] == rows.pair_id.tolist()
        model = d.build_model(cell["seed"])
        model.load_state_dict(payload["model_state_dict"])
        assert d.summarize(d.evaluate(model, head, rows)) == cell["train"]
        x = torch.tensor(rows.conditioned, device="cuda")
        y = torch.tensor(rows.target_normalized, device="cuda")
        opt = torch.optim.LBFGS(model.parameters(), **cfg["optimizer"])
        calls = []
        tick = time.perf_counter()
        def closure():
            opt.zero_grad(set_to_none=True)
            raw = ((d.predict(model, x, head)-y)**2).sum(-1).mean()
            loss = cfg["loss_scale"]*raw
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            calls.append(float(raw.detach()))
            return loss
        opt.step(closure)
        metrics = dict(train=d.evaluate(model, head, rows), validation=d.evaluate(model, head, val))
        weight = path.with_name(path.stem + '-refined.pt')
        if weight.exists():
            raise FileExistsError(weight)
        torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        source_sha256=d.sha(path), config_sha256=d.sha(cfgpath)), weight)
        rawpath = weight.with_suffix('.json')
        d.write_json(rawpath, metrics)
        with torch.no_grad():
            loss = float(((d.predict(model, x, head)-y)**2).sum(-1).mean())
        record = dict(name=cell["name"], train=d.summarize(metrics["train"]), validation=d.summarize(metrics["validation"]),
            common_local=d.summarize(d.evaluate(model, head, common)), prior_train=cell["train"]["profile_a"],
            q_loss=loss, closure_losses=calls, iterations=opt.state[next(iter(model.parameters()))]["n_iter"],
            wall_s=time.perf_counter()-tick, source_sha256=d.sha(path),
            checkpoint=dict(path=str(weight.relative_to(d.ROOT)), sha256=d.sha(weight)),
            raw=dict(path=str(rawpath.relative_to(d.ROOT)), sha256=d.sha(rawpath)))
        d.write_json(out / (cell["name"] + '.json'), record)
        results.append(record)
        print(f"{cell['name']}: train {record['prior_train']} -> {record['train']['profile_a']}/512; validation {record['validation']['profile_a']}/3600", flush=True)
    d.write_json(out / "results.json", dict(status="COMPLETE_DIAGNOSIS", results=results, config_sha256=d.sha(cfgpath),
                 interpretation=cfg["interpretation"], final_test="NOT_CREATED"))


if __name__ == "__main__":
    main()
