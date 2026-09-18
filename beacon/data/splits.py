from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Set, Sequence, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class FoldSplit:
    fold: int
    train: List[int]
    val: List[int]
    test: List[int]


def cluster_members(clusters: Sequence[str]) -> Dict[str, List[int]]:
    """Groups row indices by their cluster identifier."""
    members: Dict[str, List[int]] = {}
    for position, cluster in enumerate(clusters):
        members.setdefault(str(cluster), []).append(position)
    return members


def balanced_partition(
    members: Dict[str, List[int]], keys: List[str], bins: int, rng: np.random.Generator
) -> List[List[str]]:
    """Partitions cluster keys into balanced bins according to member counts."""
    priority = {key: float(val) for key, val in zip(keys, rng.random(len(keys)))}
    order = sorted(keys, key=lambda k: (-len(members[k]), priority[k]))
    buckets: List[List[str]] = [[] for _ in range(bins)]
    loads = [0] * bins
    for key in order:
        target = min(range(bins), key=lambda idx: (loads[idx], idx))
        buckets[target].append(key)
        loads[target] += len(members[key])
    return buckets


def take_until(
    members: Dict[str, List[int]], keys: List[str], target: int, rng: np.random.Generator
) -> List[str]:
    """Randomly selects clusters until the accumulated pair count reaches target."""
    shuffled = list(keys)
    rng.shuffle(shuffled)
    chosen: List[str] = []
    total = 0
    for key in shuffled:
        if total >= target:
            break
        chosen.append(key)
        total += len(members[key])
    return chosen


def nested_folds(
    clusters: Sequence[str],
    folds: int = 5,
    seed: int = 0,
    val_fraction: float = 0.15
) -> List[FoldSplit]:
    """
    Generates k-fold splits strictly disjoint on antigen clusters.
    Each fold reserves one bucket for test, a random fraction of remaining clusters for val,
    and all others for train.
    """
    members = cluster_members(clusters)
    rng = np.random.default_rng(seed)
    keys = list(members.keys())
    buckets = balanced_partition(members, keys, int(folds), rng)
    total = sum(len(v) for v in members.values())
    val_target = int(round(total * float(val_fraction)))

    splits: List[FoldSplit] = []
    for idx in range(int(folds)):
        test_keys = set(buckets[idx])
        remaining = [k for k in keys if k not in test_keys]
        val_keys = set(take_until(members, remaining, val_target, rng))
        train_keys = [k for k in remaining if k not in val_keys]

        test_rows = sorted(pos for k in test_keys for pos in members[k])
        val_rows = sorted(pos for k in val_keys for pos in members[k])
        train_rows = sorted(pos for k in train_keys for pos in members[k])

        split = FoldSplit(fold=idx, train=train_rows, val=val_rows, test=test_rows)
        splits.append(split)
    return splits


def assert_cluster_disjoint(split: FoldSplit, clusters: Sequence[str]) -> None:
    """Verifies that no cluster is shared between train, validation, or test sets."""
    train_c = {str(clusters[p]) for p in split.train}
    val_c = {str(clusters[p]) for p in split.val}
    test_c = {str(clusters[p]) for p in split.test}

    overlap = (train_c & val_c) | (train_c & test_c) | (val_c & test_c)
    if overlap:
        raise ValueError(f"Cluster leakage detected across sets: {sorted(list(overlap))[:5]}")


def split_summary(split: FoldSplit, total: int) -> Dict[str, float]:
    """Returns the proportional split ratios."""
    denom = float(max(total, 1))
    return {
        "train": len(split.train) / denom,
        "val": len(split.val) / denom,
        "test": len(split.test) / denom,
    }
