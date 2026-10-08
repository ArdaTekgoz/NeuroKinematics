"""Independent forward/FD gate for the opt-in FK extension; never trains."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import pinocchio as pin
from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.neural.training_fk import TrainingFK, DOMAIN
from neurokinematics.neural.physics import PhysicsLoss
from neurokinematics.neural.c104 import write_json, sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    inputs = load_robot(); oracle = PinocchioFK(inputs)
    fk = TrainingFK.from_frozen(domain=DOMAIN); loss = PhysicsLoss()
    lower, upper = np.asarray(inputs.limits).T; span = upper-lower
    target_q = lower + .37 * span

    def reference(q):
        pq = np.zeros(oracle.model.nq)
        for j, name in enumerate(inputs.joint_names): pq[oracle.q_indices[name]] = q[j]
        pin.forwardKinematics(oracle.model, oracle.data, pq); pin.updateFramePlacements(oracle.model, oracle.data)
        base = oracle.data.oMf[oracle.frame_ids[inputs.base]].homogeneous.copy()
        tcp = oracle.data.oMf[oracle.frame_ids[inputs.tcp]].homogeneous.copy()
        return np.linalg.inv(base) @ tcp

    desired = reference(target_q)
    target_n = torch.tensor((target_q-lower)/span, dtype=torch.float64)
    pdes = torch.tensor(desired[:3,3]); rdes = torch.tensor(desired[:3,:3])

    def objective(q):
        t = fk(q, robot_id=fk.robot_id, joint_names=fk.joint_names)
        parts, _, _ = loss.components((q-torch.tensor(lower))/torch.tensor(span), target_n, pdes, rdes)
        return torch.cat((t[:3,:].reshape(-1), torch.stack([parts[k] for k in ('q','p','R')]),
                          (parts['q']+parts['p']+parts['R']).reshape(1)))

    def reference_objective(q):
        t = reference(q)
        lq = np.sum(((q-target_q)/span)**2)
        lp = np.sum(((t[:3,3]-desired[:3,3])/.9015)**2)
        lr = np.sum((t[:3,:3]-desired[:3,:3])**2)/8
        return np.r_[t[:3,:].reshape(-1), lq, lp, lr, lq+lp+lr]

    samples = json.loads((ROOT/'experiments/C1-05/fk-domain-samples.json').read_text())['samples']
    maxima = dict(position64=0., rotation64=0., position32=0., rotation32=0., gradient_abs=0., tolerance_ratio=0.)
    raw = a.output/'results.jsonl'; passed = True
    with raw.open('w',encoding='utf-8',newline='\n') as stream:
        for sample in samples:
            q = np.asarray(sample['q_rad'])
            for dtype, tag, tol in [(torch.float64,'64',1e-9),(torch.float32,'32',1e-5)]:
                qt = torch.tensor(q,dtype=dtype)
                actual = fk(qt,robot_id=fk.robot_id,joint_names=fk.joint_names).double().numpy()
                expected = reference(qt.double().numpy())
                pe = float(np.linalg.norm(actual[:3,3]-expected[:3,3])); re = float(np.linalg.norm(actual[:3,:3]-expected[:3,:3]))
                maxima['position'+tag]=max(maxima['position'+tag],pe); maxima['rotation'+tag]=max(maxima['rotation'+tag],re)
                ok = pe<=tol and re<=tol
                stream.write(json.dumps(dict(id=sample['id'],kind=sample['kind'],test='forward'+tag,q=qt.tolist(),position_m=pe,rotation_fro=re,PASS=ok))+'\n')
                passed &= ok
            qt = torch.tensor(q,dtype=torch.float64,requires_grad=True)
            ag = torch.autograd.functional.jacobian(objective,qt).numpy()
            fd = np.empty_like(ag)
            for j in range(6):
                offset=np.zeros(6); offset[j]=1e-6
                fd[:,j]=(reference_objective(q+offset)-reference_objective(q-offset))/(2e-6)
            error=np.abs(ag-fd); ratio=error/(1e-5+1e-3*np.abs(fd))
            gc = torch.autograd.gradcheck(objective,(qt,),eps=1e-6,atol=1e-5,rtol=1e-3,raise_exception=False)
            ok=bool(np.all(ratio<=1) and gc and np.isfinite(ag).all())
            maxima['gradient_abs']=max(maxima['gradient_abs'],float(error.max())); maxima['tolerance_ratio']=max(maxima['tolerance_ratio'],float(ratio.max()))
            stream.write(json.dumps(dict(id=sample['id'],kind=sample['kind'],test='gradient',q=q.tolist(),autograd=ag.tolist(),finite_difference=fd.tolist(),epsilon=1e-6,gradcheck=gc,PASS=ok))+'\n')
            passed &= ok
    write_json(a.output/'summary.json',dict(status='PASS' if passed else 'FAIL',samples=len(samples),forward_checks=160,gradient_checks=80,gradchecks=80,maxima=maxima,raw_sha256=sha(raw),
        sources={p:sha(ROOT/p) for p in ['src/neurokinematics/neural/training_fk.py','src/neurokinematics/neural/physics.py']},
        sample_sha256=sha(ROOT/'experiments/C1-05/fk-domain-samples.json')))
    print(json.dumps(dict(status='PASS' if passed else 'FAIL',maxima=maxima)))
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
