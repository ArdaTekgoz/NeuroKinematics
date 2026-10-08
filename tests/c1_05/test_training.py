import dataclasses
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from neurokinematics.neural import c105
from neurokinematics.neural.c104 import load_data, validate_rows, MLP
from neurokinematics.neural.physics import PhysicsLoss


def test_inventory_and_no_leakage(monkeypatch):
    from neurokinematics.data import factory
    from neurokinematics.neural import c104
    seen=[]; original=c104.read_shard
    def read(path,*args,**kwargs):
        assert '-test-' not in str(path)
        seen.append(str(path)); return original(path,*args,**kwargs)
    monkeypatch.setattr(c104,'read_shard',read)
    train,val=load_data(label_fk=False)
    assert len(val.pair_id)==3600 and sum(~val.label_present)==351 and len(set(val.pair_id))==3600
    assert sum(train.label_present)==15204 and seen
    changed=dataclasses.replace(train,q_target=np.roll(train.q_target,1,axis=0))
    assert np.array_equal(changed.conditioned,train.conditioned)
    with pytest.raises(ValueError): validate_rows(changed)
    leaked=dataclasses.replace(train,conditioned=np.c_[train.pose_only,train.target_normalized])
    with pytest.raises(ValueError): validate_rows(leaked)


def test_control_gradients_and_selection():
    torch.manual_seed(2026100201); model=MLP('conditioned')
    train,_=load_data(label_fk=False); rows=train.take(np.flatnonzero(train.label_present)[:32]); data=c105.prepared(rows)
    loss,parts,z,q,t,logits=c105.objective(model,'Q',data,np.arange(32),PhysicsLoss())
    direct=((model(data['x'])-data['y'])**2).sum(-1).mean()
    g1=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    g2=torch.autograd.grad(direct,tuple(model.parameters()))
    assert torch.equal(loss,direct) and all(torch.equal(a,b) for a,b in zip(g1,g2))
    record=c105.gradient_record(model,parts,z,logits,c105.configuration('Q'),PhysicsLoss())
    assert record['p']['raw_q_output_norm']>0 and record['R']['raw_q_output_norm']>0


def test_matrix_and_denominator():
    assert c105.PAIRS=={'E-C03':('Q','FK'),'E-C04':('FK','FK_LIMIT'),'E-C05':('FK','FK_TANH')}
    assert c105.configuration('FK_TANH')['lambda_lim']==0
    assert c105.configuration('FK_LIMIT')['head']=='unbounded'
    quant=c105.full_quantiles([1.,2.],4)
    assert quant['n']==4 and quant['median']==2. and quant['p95'] is None and quant['p95_unbounded']
    assert c105.full_quantiles([1.,2.],2)['median']==1.


def test_checkpoint_integrity(tmp_path):
    model=MLP('conditioned'); opt=torch.optim.AdamW(model.parameters())
    path=tmp_path/'best.pt'; c105.save_checkpoint(path,model,opt,'FK',2026100201,2,.4,'E-C03')
    loaded,metadata=c105.load_checkpoint(path)
    assert c105.tensor_hash(model)==c105.tensor_hash(loaded)
    payload=torch.load(path,weights_only=True); payload['config_sha256']='0'*64; torch.save(payload,path)
    with pytest.raises(ValueError,match='metadata'): c105.load_checkpoint(path)


def test_evaluation_keeps_invalid(monkeypatch,tmp_path):
    _,val=load_data(label_fk=False); ids=np.r_[np.flatnonzero(val.label_present)[:2],np.flatnonzero(~val.label_present)[:2]]
    rows=val.take(ids); raw=rows.q_current.copy(); raw[0,:]=100.; raw[1,:]=np.nan
    monkeypatch.setattr(c105,'infer',lambda *a:(raw,np.zeros((4,6),dtype=np.float32)))
    summary=c105.evaluate(None,'FK',rows,tmp_path/'rows.jsonl')
    overall=summary['breakdowns']['overall']
    assert overall['n']==4 and overall['labeled']==2 and overall['valid_raw']==2
    assert overall['nonfinite']==1 and overall['out_of_limits']==1
    assert overall['profile_a']==0 and overall['position_m_full']['p95_unbounded']
