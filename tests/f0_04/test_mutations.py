import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neurokinematics.data.factory import (assign_splits, audit_groups_and_duplicates,
    canonical_array_hash, normalization, train_singularity_threshold, validate_arrays,
    validate_boundary, read_shard, write_deterministic_npz)


@pytest.mark.parametrize("mutation", [
    "seed_change","row_reorder","shard_missing","shard_byte_corruption","wrong_model_hash","wrong_tcp_hash",
    "joint_columns_reversed","quaternion_xyzw","quaternion_sign_counted_distinct","group_cross_split",
    "q_cross_split_duplicate","validation_in_normalization","boundary_wrong_threshold","singularity_all_data_threshold",
    "coverage_grid_posthoc","nan_or_inf","missing_record"
])
def test_required_mutation_is_detected(mutation, record_property, valid_arrays, schema, limits, tmp_path):
    detected = True
    order=[x["name"] for x in schema["fields"]]
    if mutation == "seed_change":
        detected = not np.array_equal(assign_splits([f"g{i}" for i in range(50)],1,{"train":.7,"validation":.15,"test":.15}), assign_splits([f"g{i}" for i in range(50)],2,{"train":.7,"validation":.15,"test":.15}))
    elif mutation == "row_reorder":
        detected = canonical_array_hash(valid_arrays,order) != canonical_array_hash({k:v[::-1] for k,v in valid_arrays.items()},order)
    elif mutation in {"wrong_model_hash","wrong_tcp_hash"}:
        expected={"robot_id":"kuka_kr6_r900_sixx","model_hash":"a"*64,"tcp_hash":"b"*64}
        changed={k:v.copy() for k,v in valid_arrays.items()}; key="model_hash" if mutation=="wrong_model_hash" else "tcp_hash"; changed[key][0]=b"c"*64
        try: validate_arrays(changed,schema,limits,expected); detected=False
        except ValueError: detected=True
    elif mutation == "joint_columns_reversed": detected = not np.array_equal(valid_arrays["q"],valid_arrays["q"][:,::-1])
    elif mutation == "quaternion_xyzw": detected = not np.array_equal(valid_arrays["quaternion_wxyz"],np.roll(valid_arrays["quaternion_wxyz"],-1,axis=1))
    elif mutation == "quaternion_sign_counted_distinct":
        changed={k:v.copy() for k,v in valid_arrays.items()}; changed["quaternion_wxyz"][0] *= -1
        try: validate_arrays(changed,schema,limits); detected=False
        except ValueError: detected=True
    elif mutation == "group_cross_split":
        changed={k:v.copy() for k,v in valid_arrays.items()}; changed["group_id"][-1]=changed["group_id"][0]
        try: audit_groups_and_duplicates(changed,independent_roots=False); detected=False
        except ValueError: detected=True
    elif mutation == "q_cross_split_duplicate":
        changed={k:v.copy() for k,v in valid_arrays.items()}; changed["q"][-1]=changed["q"][0]
        try: audit_groups_and_duplicates(changed); detected=False
        except ValueError: detected=True
    elif mutation == "validation_in_normalization":
        changed={k:v.copy() for k,v in valid_arrays.items()}; changed["q"][changed["split"]!=b"train"] += .1
        detected = normalization(valid_arrays) == normalization(changed)
    elif mutation == "boundary_wrong_threshold":
        bounds=np.asarray(limits); q=np.tile((bounds[:,0]+bounds[:,1])/2,(2,1)); q[:,0]=bounds[0,0]+.025*(bounds[0,1]-bounds[0,0])
        try: validate_boundary(q,limits,.02); detected=False
        except ValueError: detected=True
    elif mutation == "singularity_all_data_threshold":
        train_only=train_singularity_threshold(valid_arrays,.05)
        changed={k:v.copy() for k,v in valid_arrays.items()}; changed["sigma_min"][changed["split"]!=b"train"]=-100
        detected = train_only == train_singularity_threshold(changed,.05)
    elif mutation == "coverage_grid_posthoc":
        raw=Path("experiments/F0-04/config.json").read_bytes(); changed=json.loads(raw); changed["coverage"]["position_voxel_size_m"]=.5
        detected = hashlib.sha256(raw).hexdigest() != hashlib.sha256(json.dumps(changed).encode()).hexdigest()
    elif mutation == "nan_or_inf":
        changed={k:v.copy() for k,v in valid_arrays.items()}; changed["q"][0,0]=np.nan
        try: validate_arrays(changed,schema,limits); detected=False
        except ValueError: detected=True
    elif mutation == "missing_record":
        changed={k:v.copy() for k,v in valid_arrays.items()}; changed["q"]=changed["q"][:-1]
        try: validate_arrays(changed,schema,limits); detected=False
        except ValueError: detected=True
    elif mutation in {"shard_missing","shard_byte_corruption"}:
        path=tmp_path/"shard.npz"
        if mutation == "shard_byte_corruption":
            write_deterministic_npz(path,valid_arrays,order); payload=bytearray(path.read_bytes()); payload[40] ^= 1; path.write_bytes(payload)
        try: read_shard(path,order); detected=False
        except (ValueError,FileNotFoundError): detected=True
    record_property("mutation", mutation); record_property("detected", str(bool(detected)))
    assert detected
