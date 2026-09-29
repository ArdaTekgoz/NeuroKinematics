from copy import deepcopy
import json
from pathlib import Path
import shutil
import pytest
from neurokinematics.core import torch_validation as validation
from neurokinematics.kinematics.model import ROOT

@pytest.mark.parametrize('name',['config.json','samples.jsonl'])
def test_M18_contract_hash(name,monkeypatch):
    path=ROOT/'experiments/C1-03'/name
    original=Path.read_bytes
    def changed(self):
        raw=original(self)
        return raw+b' ' if self==path else raw
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(ValueError,match='Stage1 hash mismatch'):validation.load_contract()

@pytest.mark.parametrize('fault',['missing','duplicate','dtype','failure','hash','q','epsilon','stencil'])
def test_N24_evidence_rejects_corruption(tmp_path,fault):
    config,rows=validation.load_contract()
    source=ROOT/'experiments/C1-03/stage2/smoke-final-a'
    # Fixture is an actual measured smoke artifact, not a fabricated passing oracle.
    validation.verify_evidence(source,config,rows,smoke=True)
    target=tmp_path/'evidence';shutil.copytree(source,target)
    records=[json.loads(s) for s in (target/'results.jsonl').read_text().splitlines()]
    if fault=='missing':records.pop(0)
    elif fault=='duplicate':records.append(deepcopy(records[0]))
    elif fault=='dtype':records[0]['dtype']='float32'
    elif fault=='q':records[0]['q'][0]+=.1
    elif fault=='epsilon':next(r for r in records if r['kind']=='gradient')['epsilon']=1e-4
    elif fault=='stencil':next(r for r in records if r['kind']=='gradient')['stencils'][0]='forward2'
    elif fault=='failure':(target/'failures.jsonl').write_text('{"passed":false}\n')
    elif fault=='hash':(target/'environment.json').write_text('{}')
    (target/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8',newline='\n')
    if fault!='hash':
        hashes=json.loads((target/'hashes.json').read_text())
        for name in hashes:hashes[name]=validation.sha((target/name).read_bytes())
        validation.write_json(target/'hashes.json',hashes)
    with pytest.raises(ValueError):validation.verify_evidence(target,config,rows,smoke=True)
