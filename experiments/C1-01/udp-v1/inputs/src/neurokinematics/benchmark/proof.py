"""Conservative analytic outer reach certificate; never a solver inference."""

import hashlib
import json
import numpy as np

from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.metrics import finite_array
from neurokinematics.kinematics.model import load_robot
from neurokinematics.solvers.dls import SolverStatus


def prove_outer_reach(target_position_m, inputs=None):
    inputs=load_robot() if inputs is None else inputs
    target=finite_array(target_position_m,(3,))
    length=sum(float(np.linalg.norm(j.origin[:3,3])) for j in IndependentFK(inputs).chain)
    norm=float(np.linalg.norm(target))
    tolerance=.002; margin=1e-12
    if not norm>length+tolerance+margin:return None
    certificate={'kind':'TRIANGLE_INEQUALITY_OUTER_POSITION',
      'robot_id':inputs.robot_id,'urdf_sha256':hashlib.sha256(inputs.urdf).hexdigest(),
      'base':inputs.base,'tcp':inputs.tcp,'joint_order':list(inputs.joint_names),
      'target_position_m':target.tolist(),'target_distance_m':norm,
      'sum_origin_translation_norm_m':length,'profile_a_position_tolerance_m':tolerance,
      'roundoff_margin_m':margin,'inequality':'target_distance_m > sum_origin_translation_norm_m + profile_a_position_tolerance_m + roundoff_margin_m'}
    raw=json.dumps(certificate,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return certificate,hashlib.sha256(raw).hexdigest()


def verify_outer_reach(certificate,digest,inputs=None):
    try:
        expected=prove_outer_reach(certificate['target_position_m'],inputs)
    except (KeyError,ValueError,TypeError):return False
    return expected is not None and certificate==expected[0] and digest==expected[1]


def classify_outer_target(target_position_m, inputs=None):
    """Emit PROVEN_UNREACHABLE only with a reverified analytic certificate."""
    result=prove_outer_reach(target_position_m,inputs)
    if result is None:return SolverStatus.UNRESOLVED,None
    if not verify_outer_reach(*result,inputs):
        raise ValueError('outer reach proof failed independent verification')
    return SolverStatus.PROVEN_UNREACHABLE,result
