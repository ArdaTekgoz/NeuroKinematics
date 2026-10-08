"""C1-06 preregistered evaluation primitives; no dataset opening or model selection."""
from collections import defaultdict
import math
import time
import numpy as np

from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.metrics import quaternion_rotation, rotation_error
from neurokinematics.kinematics.model import load_robot

SEEDS = (2026100201, 2026100202, 2026100203)
SUBSETS = ('main', 'boundary', 'singularity')
BOOTSTRAP_SEED = 2026100806
BOOTSTRAP_REPLICATES = 10000


def features(query, normalization, limits):
    """Explicit input whitelist: labels/source q never enter the model."""
    position = np.asarray(query['target_position_m'], dtype=np.float64)
    quat = np.asarray(query['target_quaternion_wxyz'], dtype=np.float64)
    current = np.asarray(query['q_current'], dtype=np.float64)
    lower, upper = np.asarray(limits, dtype=np.float64).T
    if position.shape != (3,) or quat.shape != (4,) or current.shape != (6,):
        raise ValueError('input shape')
    if not all(np.isfinite(a).all() for a in (position, quat, current)):
        raise ValueError('nonfinite input')
    if abs(np.linalg.norm(quat) - 1) > 1e-12 or quat[0] < 0:
        raise ValueError('quaternion contract')
    if np.any(current < lower) or np.any(current > upper):
        raise ValueError('current limits')
    mean = np.asarray(normalization['position_mean_m'])
    std = np.asarray(normalization['position_std_m'])
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('normalization')
    return np.concatenate(((position - mean) / std, quat, (current - lower) / (upper - lower))).astype(np.float32)


class Validator:
    def __init__(self):
        self.robot = load_robot()
        self.lower, self.upper = np.asarray(self.robot.limits).T
        self.fk = IndependentFK(self.robot)

    def check(self, q, query):
        q = np.asarray(q, dtype=np.float64)
        shape = q.shape == (6,)
        finite = bool(shape and np.isfinite(q).all())
        limits = bool(finite and np.all(q >= self.lower) and np.all(q <= self.upper))
        pe = re = None
        if limits:
            pose = self.fk.forward_kinematics(q)
            pe = float(np.linalg.norm(pose[:3, 3] - np.asarray(query['target_position_m'])))
            re = math.degrees(rotation_error(pose[:3, :3], quaternion_rotation(query['target_quaternion_wxyz'])))
        a = bool(limits and pe <= .002 and re <= 1.)
        b = bool(limits and pe <= .001 and re <= .5)
        category = ('INVALID_SHAPE' if not shape else 'NONFINITE' if not finite else
                    'JOINT_LIMIT' if not limits else 'SUCCESS' if a else 'POSE_TOLERANCE')
        return dict(finite=finite, in_limits=limits, profile_a=a, profile_b=b,
                    position_error_m=pe, orientation_error_deg=re,
                    failure_class=category, collision='NOT_CHECKED')


def timed_query(query, predict, normalization, validator):
    """CPU single query: input preparation + prediction + independent validation."""
    start = time.perf_counter_ns()
    x = features(query, normalization, validator.robot.limits)
    q = predict(x)
    result = validator.check(q, query)
    elapsed = time.perf_counter_ns() - start
    return dict(result, elapsed_ns=elapsed, timeout_10ms=elapsed > 10000000,
                timeout_50ms=elapsed > 50000000)


def paired_bootstrap(candidate, control, *, seeds=SEEDS, repeats=BOOTSTRAP_REPLICATES, rng_seed=BOOTSTRAP_SEED):
    """One boolean per query/seed. Joint resampling preserves cross-subset roots.

    Rows: query_id, group_id, subsets (memberships), seed, success.
    Five timing passes MUST be reduced before this API; duplicates are rejected.
    """
    if repeats < 2 or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('bootstrap configuration')

    def index(rows):
        out = {}
        for row in rows:
            key = (row['seed'], row['query_id'])
            memberships = tuple(sorted(set(row['subsets'])))
            if key in out or row['seed'] not in seeds or type(row['success']) is not bool:
                raise ValueError('duplicate, seed or non-binary observation')
            if not memberships or not set(memberships) <= set(SUBSETS) or not row['group_id']:
                raise ValueError('group/membership')
            out[key] = (row['group_id'], memberships, int(row['success']))
        return out

    left, right = index(candidate), index(control)
    if not left or left.keys() != right.keys():
        raise ValueError('missing/unpaired observations')
    ids = sorted(q for s, q in left if s == seeds[0])
    if any({q for s, q in left if s == seed} != set(ids) for seed in seeds):
        raise ValueError('unequal seed inventory')
    groups = sorted({left[(seeds[0], q)][0] for q in ids})
    gi = {g: i for i, g in enumerate(groups)}
    sums = np.zeros((len(groups), len(seeds), 3))
    counts = np.zeros((len(groups), 3))
    for q in ids:
        base = left[(seeds[0], q)][:2]
        for sindex, seed in enumerate(seeds):
            a, b = left[(seed, q)], right[(seed, q)]
            if a[:2] != base or b[:2] != base:
                raise ValueError('query group/membership mismatch')
            for subset in a[1]:
                j = SUBSETS.index(subset)
                sums[gi[a[0]], sindex, j] += a[2] - b[2]
                if sindex == 0:
                    counts[gi[a[0]], j] += 1
    if np.any(counts.sum(axis=0) == 0):
        raise ValueError('missing subset')
    # Stratify whole roots by their union of memberships, never split a root.
    strata = defaultdict(list)
    for i in range(len(groups)):
        strata[tuple(counts[i] > 0)].append(i)
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    boot = np.empty((repeats, len(seeds), 4))
    for b in range(repeats):
        chosen = np.concatenate([rng.choice(v, size=len(v), replace=True) for _, v in sorted(strata.items())])
        values = sums[chosen].sum(axis=0) / counts[chosen].sum(axis=0)
        boot[b, :, :3] = values
        boot[b, :, 3] = .5 * (values[:, 1] + values[:, 2])
    point = sums.sum(axis=0) / counts.sum(axis=0)
    point = np.column_stack((point, .5 * (point[:, 1] + point[:, 2])))

    def summary(p, samples):
        return {name: dict(difference=float(p[j]), ci95=np.percentile(samples[:, j], [2.5, 97.5]).tolist())
                for j, name in enumerate((*SUBSETS, 'hard_equal_weight'))}

    return dict(unique_queries=len(ids), root_groups=len(groups),
                subset_n=dict(zip(SUBSETS, counts.sum(axis=0).astype(int).tolist())),
                per_seed={str(s): summary(point[i], boot[:, i]) for i, s in enumerate(seeds)},
                mean_over_fixed_seeds=summary(point.mean(axis=0), boot.mean(axis=1)),
                seed_range={name: [float(point[:, j].min()), float(point[:, j].max())]
                            for j, name in enumerate((*SUBSETS, 'hard_equal_weight'))},
                ci_scope='query-root uncertainty conditional on these three fixed training seeds',
                bootstrap_seed=rng_seed, bootstrap_replicates=repeats)


def h2_decision(result, *, integrity=True):
    if not integrity:
        return 'INCONCLUSIVE_TECHNICAL'
    stats = result['mean_over_fixed_seeds']
    hard, main = stats['hard_equal_weight'], stats['main']
    for row in (hard, main):
        lo, hi = row['ci95']
        if not all(np.isfinite([lo, hi, row['difference']])) or not -1 <= lo <= hi <= 1 or not -1 <= row['difference'] <= 1:
            raise ValueError('invalid interval')
    if hard['ci95'][0] >= .02 and main['ci95'][0] >= -.01:
        return 'SUPPORTED'
    if hard['ci95'][1] < .02 or main['ci95'][1] < -.01:
        return 'REJECTED'
    return 'INCONCLUSIVE'
