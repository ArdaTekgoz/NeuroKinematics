"""Post-hoc error decomposition of fixed checkpoints; no model selection/training."""
from pathlib import Path
import numpy as np
from neurokinematics.neural import c106r_diagnostic4 as e


def partition(records):
    groups = dict(pass_a=0, position_only_fails=0, orientation_only_fails=0, both_fail=0, invalid=0)
    for x in records:
        if not x['valid']:
            groups['invalid'] += 1
        elif x['position_m'] <= .002 and x['orientation_deg'] <= 1:
            groups['pass_a'] += 1
        elif x['position_m'] > .002 and x['orientation_deg'] <= 1:
            groups['position_only_fails'] += 1
        elif x['position_m'] <= .002 and x['orientation_deg'] > 1:
            groups['orientation_only_fails'] += 1
        else:
            groups['both_fail'] += 1
    assert sum(groups.values()) == len(records)
    return dict(n=len(records), **groups)


def main():
    d = e.d
    output = e.BASE/'error-decomposition.json'
    if output.exists():
        raise FileExistsError(output)
    result = d.read_json(e.BASE/'results.json')
    assert d.read_json(e.BASE/'audit.json')['status'] == 'PASS'
    d.guard_hashes(result['inputs'])
    train, val = d.load_data(label_fk=True)
    rows = d.matched_rows(train,2048)[0]['local']
    robot = e.r.load_robot()
    baseline = d.geometric_metrics(rows.q_current, rows, details=True)
    correction = rows.q_target - rows.q_current
    cells = []
    for cell in result['cells']:
        raw_path = d.ROOT/cell['raw']['path']
        assert d.sha(raw_path) == cell['raw']['sha256']
        raw = d.read_json(raw_path)
        records = raw['train']['rows']
        assert [x['pair_id'] for x in records] == rows.pair_id.tolist()
        assert [x['pair_id'] for x in raw['validation']['rows']] == val.pair_id.tolist()
        q = np.asarray([x['q_rad'] for x in records])
        assert np.isfinite(q).all()
        error = q - rows.q_target
        cells.append(dict(name=cell['name'],train=partition(records),
            local_validation=partition([x for x,m in zip(raw['validation']['rows'],val.mode) if m=='local']),
            wide_validation=partition([x for x,m in zip(raw['validation']['rows'],val.mode) if m=='wide']),
            train_joint_rmse_deg=np.rad2deg(np.sqrt((error**2).mean(0))).tolist(),
            train_joint_bias_deg=np.rad2deg(error.mean(0)).tolist(),
            train_joint_mse_vs_current=((error**2).mean(0)/(correction**2).mean(0)).tolist(),
            train_joint_limit_violations=((q < robot.limits[:,0]) | (q > robot.limits[:,1])).sum(0).tolist()))
    d.write_json(output,dict(status='COMPLETE_POST_HOC_DESCRIPTIVE_ANALYSIS',
        source_results_sha256=d.sha(e.BASE/'results.json'), code_sha256=d.sha(Path(__file__)),
        selection_or_training='NONE', joint_order=list(robot.joint_names),
        train_current_baseline=partition(baseline['rows']),
        teacher_correction_rmse_deg=np.rad2deg(np.sqrt((correction**2).mean(0))).tolist(),
        cells=cells, interpretation='descriptive only; error partition does not establish a causal root cause',
        final_test='NOT_CREATED',old_final_raw='NOT_READ'))
    for cell in cells:
        print(cell['name'],cell['train'],cell['local_validation'])


if __name__ == '__main__':
    main()
