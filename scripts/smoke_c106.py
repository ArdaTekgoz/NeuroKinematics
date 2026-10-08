"""Synthetic analytical smoke; does not load a model or any final query file."""
import json
from pathlib import Path
import numpy as np
from neurokinematics.neural.c106 import Validator, timed_query
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.data.factory import canonical_quaternion

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'experiments/C1-06/synthetic-smoke.jsonl'
validator = Validator()
norm = json.loads((ROOT/'experiments/C1-02/normalization.json').read_bytes())
oracle = PinocchioFK(validator.robot)
records = []
for i in range(10):
    q = validator.lower + (.2 + i * .06) * (validator.upper - validator.lower)
    pose = oracle.reference_forward_kinematics(q)
    query = dict(query_id=f'synthetic-{i:02d}', target_position_m=pose[:3, 3].tolist(),
                 target_quaternion_wxyz=canonical_quaternion(pose[:3, :3]).tolist(), q_current=q.tolist())
    value = q.copy()
    if i == 7: value[0] = np.nan
    if i == 8: value[0] = validator.upper[0] + .01
    if i == 9: query['target_position_m'][0] += .1
    records.append(dict(query_id=query['query_id'], **timed_query(query, lambda x: value, norm, validator)))
assert len(records) == 10 and sum(r['profile_a'] for r in records) == 7
assert records[7]['failure_class'] == 'NONFINITE' and records[8]['failure_class'] == 'JOINT_LIMIT'
with out.open('x', encoding='utf-8', newline='\n') as stream:
    for row in records:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
print(json.dumps(dict(status='PASS', n=10, success=7, model_inference='NOT_RUN', final_test='SEALED_NOT_RUN')))
