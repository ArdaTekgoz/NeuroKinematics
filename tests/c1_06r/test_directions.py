from dataclasses import replace
import numpy as np
import pytest
from neurokinematics.neural import c106r_directions as f


@pytest.fixture(scope='module')
def data():
    train,val=f.d.load_data(label_fk=False)
    return f.d.matched_rows(train,8)[0]['local'],val


def test_generation_replay_root_identity_and_original_preservation(data):
    roots,val=data
    rows,meta=f.generate(roots,8,'train',2026101001)
    other,_=f.generate(roots,8,'train',2026101001)
    probe,_=f.generate(roots,8,'probe',2026101001)
    assert np.array_equal(rows.q_current,other.q_current)
    assert np.array_equal(rows.q_current[::8],roots.q_current)
    assert np.array_equal(rows.q_target,np.repeat(roots.q_target,8,axis=0))
    assert np.array_equal(rows.position,np.repeat(roots.position,8,axis=0))
    assert np.max(np.abs(rows.q_current-rows.q_target))<=.1+1e-15
    assert sum(x['original'] for x in meta)==8
    f.check_separation(roots,rows,probe,val)


def test_limit_rejection_without_clipping_and_role_seed_separation():
    bounds=np.tile([-1.,1.],(6,1));q=bounds[:,1].copy()
    value,attempts=f.perturb(q,bounds,42)
    assert attempts>1 and np.all(value<1) and np.all(value>0.9)
    assert f.direction_seed(1,'root','train',1)!=f.direction_seed(1,'root','probe',1)


def test_leakage_overlap_and_nontrain_mutants_are_rejected(data):
    roots,val=data
    rows,_=f.generate(roots,8,'train',1);probe,_=f.generate(roots,8,'probe',1)
    with pytest.raises(ValueError):f.generate(replace(roots,split='validation'),8,'train',1)
    with pytest.raises(ValueError):f.check_separation(roots,rows,rows,val)
    with pytest.raises(ValueError):f.check_separation(roots,rows,probe,roots)
    with pytest.raises(ValueError):f.generate(replace(roots,label_present=np.zeros(8,dtype=bool)),8,'train',1)


def test_new_inputs_reconstruct_target_without_label_features(data):
    roots,_=data;rows,_=f.generate(roots,8,'probe',1)
    rel=f.r.relative_pose(rows);fk=f.s.IndependentFK(f.r.load_robot())
    for i,q in enumerate(rows.q_current):
        t=fk.forward_kinematics(q)
        assert np.allclose(t[:3,3]+rel[i,:3],rows.position[i],atol=1e-12,rtol=0)
        assert np.allclose(t[:3,:3]@f.r.quaternion_rotation(rel[i,3:]),f.r.quaternion_rotation(rows.quaternion[i]),atol=1e-12,rtol=0)
    changed=replace(rows,q_target=np.full_like(rows.q_target,42),target_normalized=np.zeros_like(rows.target_normalized))
    assert np.array_equal(rel,f.r.relative_pose(changed))
