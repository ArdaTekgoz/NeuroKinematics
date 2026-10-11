"""Train/validation-only oracle, branch and numerical-precision review."""
import importlib.util
from pathlib import Path
import math
import numpy as np
import torch
from neurokinematics.neural import c106r_diagnostic2 as d
from neurokinematics.kinematics.model import load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.custom_fk import IndependentFK


def main():
    d.configure()
    output = d.BASE / "project-review/pipeline.json"
    if output.exists():
        raise FileExistsError(output)
    robot = load_robot()
    pin, independent = PinocchioFK(robot), IndependentFK(robot)
    physics = d.PhysicsLoss()
    train, val = d.load_data(label_fk=True)
    spec = importlib.util.spec_from_file_location('round1_audit', d.ROOT / 'scripts/audit_c106r_round1.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    results = {}
    for rows in (train, val):
        labeled = rows.take(np.flatnonzero(rows.label_present))
        z = torch.tensor(labeled.target_normalized, device="cuda")
        q32 = physics.raw(z).cpu().numpy()
        lower, upper = np.asarray(robot.limits).T
        q64 = lower + labeled.target_normalized.astype(np.float64)*(upper-lower)
        # Perfect normalized teacher predictions traversing the production decoder.
        oracle = {"teacher_float64": d.geometric_metrics(labeled.q_target, labeled, details=True),
                  "decoded_float32": d.geometric_metrics(q32, labeled, details=True),
                  "decoded_float64": d.geometric_metrics(q64, labeled, details=True)}
        maxp = maxr = 0.
        for q in labeled.q_target:
            a, b = pin.reference_forward_kinematics(q), independent.forward_kinematics(q)
            maxp = max(maxp, float(np.linalg.norm(a[:3, 3]-b[:3, 3])))
            maxr = max(maxr, float(np.linalg.norm(a[:3, :3]-b[:3, :3])))
        assert maxp <= 1e-9 and maxr <= 1e-9
        bymode = {m: {str(rows.source_sample_id[i]): i for i in np.flatnonzero((rows.mode == m) & rows.label_present)}
                  for m in ('local', 'wide')}
        roots = sorted(set(bymode['local']) & set(bymode['wide']))
        il = np.array([bymode['local'][r] for r in roots])
        iw = np.array([bymode['wide'][r] for r in roots])
        assert np.array_equal(rows.position[il], rows.position[iw])
        assert np.array_equal(rows.quaternion[il], rows.quaternion[iw])
        diff = rows.q_target[il]-rows.q_target[iw]
        midpoint = .5*(rows.q_target[il]+rows.q_target[iw])
        midpoint_metric = d.geometric_metrics(midpoint, rows.take(il))
        reconstruction_loss = ((z-torch.tensor(labeled.target_normalized, device='cuda'))**2).sum(-1)
        assert float(reconstruction_loss.max()) == 0.
        bad32 = np.flatnonzero((q32 < lower).any(1) | (q32 > upper).any(1))
        results[rows.split] = dict(rows=len(rows.pair_id), labeled=len(labeled.pair_id),
            teacher_oracles={k:d.summarize(v) for k,v in oracle.items()},
            decoder_max_q_error_rad=float(np.max(np.abs(q32-labeled.q_target))),
            decoder_out_of_limit_rows=len(bad32), decoder_first_ids=labeled.pair_id[bad32[:10]].tolist(),
            independent_fk_max_position_m=maxp, independent_fk_max_rotation_frobenius=maxr,
            paired_pose_roots=len(roots), teacher_joint_difference_l2_rad_quantiles={str(p):float(np.quantile(np.linalg.norm(diff,axis=1),p)) for p in (.5,.95,1)},
            midpoint_pose=d.summarize(midpoint_metric),
            interpretation="Local/wide current inputs differ. Midpoint failure demonstrates nonconvex solution geometry, not contradictory identical inputs or proof that a model averages branches.",
            supervised_local_fraction=float(((rows.mode=='local') & rows.label_present).sum()/rows.label_present.sum()),
            missing_wide=int((~rows.label_present).sum()))
    # Summarize actual raw predictions by mode; retain invalids as +inf/null.
    strata = {}
    for cell in d.read_json(d.BASE / 'results.json')['cells']:
        path = d.ROOT / cell['raw']['path']
        assert d.sha(path) == cell['raw']['sha256']
        raw = d.read_json(path)
        strata[cell['name']] = audit.summary(raw['validation'], val)['groups']
    refined = {}
    for cell in d.read_json(d.BASE / 'refinement/results.json')['results']:
        path = d.ROOT / cell['raw']['path']
        assert d.sha(path) == cell['raw']['sha256']
        refined[cell['name']] = audit.summary(d.read_json(path)['validation'], val)['groups']
    d.write_json(output, dict(status="COMPLETE_REVIEW", splits=results, validation_strata=strata,
                             refined_validation_strata=refined, code_sha256=d.sha(Path(__file__)),
                             source_hashes={str(p.relative_to(d.ROOT)):d.sha(p) for p in [d.CONFIG,d.BASE/'results.json',d.BASE/'refinement/results.json']},
                             test_raw="NOT_READ", final_test="NOT_CREATED"))
    print("PASS: teacher oracles, independent FK and denominator review; see pipeline.json", flush=True)


if __name__ == '__main__':
    main()
