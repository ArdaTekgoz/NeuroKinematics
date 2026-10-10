"""Audit diagnostic5, paired loss follow-up, denominators and immutable inputs."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from neurokinematics.neural import c106r_diagnostic5 as s
from neurokinematics.neural import c106r_pose_followup as f


def main():
    d=s.d;out=s.BASE/'audit.json'
    if out.exists():raise FileExistsError(out)
    diag=d.read_json(s.BASE/'results.json');follow=d.read_json(f.BASE/'results.json')
    frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    for base,result in ((s.BASE,diag),(f.BASE,follow)):
        assert result['inputs']==d.read_json(base/'registration.json')['inputs']
        d.guard_hashes(result['inputs'])
    sampling=d.read_json(s.BASE/'sampling-review.json');d.guard_hashes(sampling['sources'])
    assert d.sha(d.ROOT/sampling['raw']['path'])==sampling['raw']['sha256']
    assert sampling['provenance_replayed']==7000
    assert diag['preflight']['status']=='PASS' and diag['preflight']['jacobian_rows']==4096
    assert diag['preflight']['fd']['n']==32
    parents=d.read_json(s.e.BASE/'results.json')['cells']
    for cell in diag['cells']:
        assert cell['replay']=='EXACT_MATCH' and cell['weights_unchanged']
        path=d.ROOT/cell['raw']['path'];assert d.sha(path)==cell['raw']['sha256']
        rows=d.read_json(path)['rows']
        assert len(rows)==len({x['pair_id'] for x in rows})==cell['n']==2048
        assert sum(x['valid'] for x in rows)==cell['valid']
        assert sum(x['profile_a'] for x in rows)==cell['profile_a']
        parent=next(c for c in parents if c['name']==cell['name'])
        old=d.read_json(d.ROOT/parent['raw']['path'])['train']['rows']
        for a,b in zip(rows,old):
            assert (a['pair_id'],a['valid'],a['profile_a'])==(b['pair_id'],b['valid'],b['profile_a'])
            if a['valid']:
                assert abs(a['position_m']-b['position_m'])<1e-9
                assert abs(a['orientation_deg']-b['orientation_deg'])<1e-7
        for groups in cell['groups'].values():
            assert sum(g['n'] for g in groups)==2048
            assert sum(g['profile_a'] for g in groups)==cell['profile_a']
        assert len(cell['gradients_batches'])==16
    assert [c['name'] for c in follow['cells']]==['Q','POSE_A']
    assert len(set(c['initial_hash'] for c in follow['cells']))==1
    assert sum(c['steps'] for c in follow['cells'])==follow['total_updates']==10000
    parent=next(c for c in parents if c['name']=='capacity')
    payload=torch.load(d.ROOT/parent['checkpoint']['path'],weights_only=True)
    model=s.e.build_model(2026100901,512,device='cpu');model.load_state_dict(payload['model_state_dict'])
    assert s.d.state_hash(model)==follow['cells'][0]['initial_hash']
    for cell in follow['cells']:
        assert cell['reload']=='EXACT_MATCH_TRAIN_VALIDATION'
        assert [h['step'] for h in cell['history']]==list(range(0,5001,1000))
        for k in ('raw','checkpoint'):assert d.sha(d.ROOT/cell[k]['path'])==cell[k]['sha256']
        saved=torch.load(d.ROOT/cell['checkpoint']['path'],weights_only=True)
        assert saved['contract']['pair_ids']==payload['contract']['pair_ids']
        assert saved['contract']['inputs']==follow['inputs'] and type(saved['contract']['torch']) is str
        raw=d.read_json(d.ROOT/cell['raw']['path'])
        for split,n in (('train',2048),('validation',3600)):
            rows=raw[split]['rows']
            assert raw[split]['n']==cell[split]['n']==len(rows)==len({x['pair_id'] for x in rows})==n
            for profile,p,a in (('profile_a',.002,1),('profile_b',.001,.5)):
                assert sum(x[profile] for x in rows)==cell[split][profile]
                for x in rows:assert x[profile]==bool(x['valid'] and x['position_m']<=p and x['orientation_deg']<=a)
    paths=list(s.BASE.glob('*-tests.xml'))+[f.BASE/'tests.xml'];counts={}
    for p in paths:
        suites=ET.parse(p).getroot().findall('testsuite')
        assert all(int(x.attrib.get(k,0))==0 for x in suites for k in ('errors','failures','skipped'))
        counts[str(p.relative_to(s.BASE))]=sum(int(x.attrib['tests']) for x in suites)
    assert sum(counts.values())==488
    d.write_json(out,dict(status='PASS',tests_passed=counts,total_tests=488,
        prior_collection_failure='commands/043 retained; six import collection errors; seven isolated processes passed',
        frozen_files=len(frozen),diagnostic_rows=4096,followup_updates=10000,followup_reload_rows=11296,
        sampling_train_rows=7000,raw_hashes='MATCH',registered_sources='MATCH',
        code_sha256=d.sha(Path(__file__)),final_test='NOT_CREATED',old_final_raw='NOT_READ'))
    print('PASS:488 tests,4096 diagnostic rows,paired10000 updates,11296 reload rows,7000 provenance checks,122 frozen inputs')


if __name__=='__main__':main()
