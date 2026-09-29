from dataclasses import replace
import ast
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import torch
from conftest import analytic_inputs
from neurokinematics.kinematics.model import FROZEN_HASHES, ROOT
from neurokinematics.kinematics.torch_fk import TorchFK, TorchSerialChain

def call(f,q,**kw):
    return f(q,robot_id=kw.get('robot_id',f.robot_id),joint_names=kw.get('joint_names',f.joint_names),units=kw.get('units','rad'))

@pytest.mark.parametrize('q',[
    [0.]*6,torch.zeros(5),torch.zeros(7),torch.zeros((0,6)),torch.zeros((2,3,6)),
    torch.zeros(6,dtype=torch.int64),torch.zeros(6,dtype=torch.float16),torch.zeros(6,dtype=torch.complex64),
    torch.full((6,),float('nan')),torch.full((6,),float('inf')),torch.full((6,),-float('inf')),torch.full((6,),90.),
    torch.zeros((2,6)).to_sparse(),
])
def test_N21_invalid(public_fk,q):
    with pytest.raises((ValueError,TypeError)): call(public_fk,q)

@pytest.mark.parametrize('kw',[{'robot_id':'other'},{'joint_names':('joint_2','joint_1','joint_3','joint_4','joint_5','joint_6')},{'units':'deg'}])
def test_N22_metadata(public_fk,kw):
    with pytest.raises(ValueError): call(public_fk,torch.zeros(6,dtype=torch.float64),**kw)

def test_N22_missing_metadata(public_fk):
    with pytest.raises(TypeError): public_fk(torch.zeros(6))

@pytest.mark.parametrize('path',list(FROZEN_HASHES))
def test_M18_robot_hash(tmp_path,path):
    for name in FROZEN_HASHES:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes()+(b' ' if name==path else b''))
    with pytest.raises(ValueError,match='hash mismatch'): TorchFK.from_frozen(tmp_path)

@pytest.mark.parametrize('fault',['missing_limit','bad_limit','missing_axis','bad_axis','unknown','prismatic','continuous','mimic','cycle','disconnected','ambiguous','missing_frame'])
def test_N23_bad_chain(fault):
    inputs=analytic_inputs('A4');xml=ET.fromstring(inputs.urdf);joint=xml.find("joint[@name='j']")
    if fault=='missing_limit':joint.remove(joint.find('limit'))
    elif fault=='bad_limit':joint.find('limit').set('lower','nan')
    elif fault=='missing_axis':joint.remove(joint.find('axis'))
    elif fault=='bad_axis':joint.find('axis').set('xyz','0 0 0')
    elif fault in ('unknown','prismatic','continuous'):joint.set('type',fault)
    elif fault=='mimic':ET.SubElement(joint,'mimic',joint='j')
    elif fault=='cycle':xml.find("joint[@name='mount']/parent").set('link','tcp')
    elif fault=='disconnected':ET.SubElement(xml,'link',name='loose')
    elif fault=='ambiguous':
        other=ET.fromstring(ET.tostring(joint));other.set('name','duplicate');xml.append(other)
    elif fault=='missing_frame':inputs=replace(inputs,base='absent')
    with pytest.raises(ValueError):TorchSerialChain(replace(inputs,urdf=ET.tostring(xml)))

def test_M20_exact_and_one_ulp(public_fk,robot):
    bounds=np.array(robot.limits);mid=bounds.mean(1)
    for j in range(6):
        for side,direction in [(0,-np.inf),(1,np.inf)]:
            q=mid.copy();q[j]=bounds[j,side]
            call(public_fk,torch.tensor(q))
            q[j]=np.nextafter(q[j],direction)
            with pytest.raises(ValueError):call(public_fk,torch.tensor(q))
            q=mid.astype('f4');q[j]=bounds[j,side]
            if bounds[j,0]<=float(q[j])<=bounds[j,1]: q[j]=np.nextafter(q[j],np.float32(direction))
            with pytest.raises(ValueError):call(public_fk,torch.tensor(q))

def test_dtype_noncontiguous_and_batch(public_fk):
    values=torch.tensor([[.3,-.6,.8,-1,.5,-.7],[.2,-.4,.6,-.8,.3,-.9]],dtype=torch.float64)
    reference=call(public_fk,values)
    # Values in columns of a wider tensor yield a noncontiguous valid input.
    wider=torch.zeros((2,12),dtype=torch.float64);wider[:,::2]=values
    assert not wider[:,::2].is_contiguous()
    assert torch.equal(call(public_fk,wider[:,::2]),reference)
    out32=call(public_fk,values.float());out64=call(public_fk,values)
    assert out32.dtype==torch.float32 and out64.dtype==torch.float64
    assert torch.equal(out64,reference)
    mixed=values.clone();mixed[1,0]=float('nan')
    with pytest.raises(ValueError):call(public_fk,mixed)

def test_graph_source_contract():
    path=ROOT/'src/neurokinematics/kinematics/torch_fk.py'
    tree=ast.parse(path.read_text())
    forwards=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__call__']
    for fn in forwards:
        for node in ast.walk(fn):
            if isinstance(node,ast.Attribute): assert node.attr not in {'numpy','detach','item','no_grad','reference_forward_kinematics'}
    assert 'pinocchio' not in path.read_text().lower()
