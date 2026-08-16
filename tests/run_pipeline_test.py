"""Quick 10K pipeline validation test — v2 (with reachable coverage fix)."""
import sys; sys.path.insert(0, '.')
from neurokinematics.data.dataset_factory import generate_dataset

results = generate_dataset(
    urdf_path='robots/kuka_kr6/kr6.urdf',
    output_path='data/kr6_test_10k_v2.h5',
    n_total=10000,
    seed=42,
    progress=True,
)

print()
print("=== MILESTONE 2 CHECKS (v2) ===")
wc = results["workspace_coverage"]
oc = results["orientation_coverage"]
jl = results["joint_leakage"]["leaking_count"]
pl = results["pose_leakage"]["leaking_count"]
fk = results["max_fk_error"]
jv = results["joint_limit_violations"]
tr = results["train_size"]
va = results["val_size"]
te = results["test_size"]
total = tr + va + te

print(f"1. Workspace Coverage (reachable): {wc:.1%}")
print(f"2. Orientation Coverage: {oc:.1%}")
print(f"3. Joint Leakage: {jl}")
print(f"4. Pose Leakage: {pl}")
print(f"5. Max FK Error: {fk:.2e} m")
print(f"6. Joint Limit Violations: {jv}")
print(f"7. Split: train={tr}({tr/total:.0%}), val={va}({va/total:.0%}), test={te}({te/total:.0%})")

# Determinism check
results2 = generate_dataset(
    urdf_path='robots/kuka_kr6/kr6.urdf',
    output_path='data/kr6_test_10k_v2b.h5',
    n_total=10000,
    seed=42,
    progress=False,
)
import h5py, numpy as np
with h5py.File('data/kr6_test_10k_v2.h5','r') as f1, h5py.File('data/kr6_test_10k_v2b.h5','r') as f2:
    q1 = f1['outputs/q'][:]
    q2 = f2['outputs/q'][:]
    det_ok = np.array_equal(q1, q2)
print(f"8. Determinism: {'PASS' if det_ok else 'FAIL'}")

all_pass = (wc >= 0.90 and jl == 0 and fk < 1e-6 and jv == 0 and det_ok)
status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS NEED ATTENTION"
print(f"\nOVERALL: {status}")
