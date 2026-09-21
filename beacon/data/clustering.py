from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

ANTIGEN_BLOCK = (
    "Antigen_ID",
    "Antigen_AA",
    "Antigen_Epitope",
    "Antigen_AA_Uniprot",
    "Antigen_UniProt_ID",
    "Cluster_id",
)

MAX_ALIGNMENT_REPRESENTATIVES = 8000


def pair_hash(vh: str, vl: str, antigen: str) -> str:
    payload = "|".join((str(vh or ""), str(vl or ""), str(antigen or ""))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class ClusterAssignment:
    split_clusters: list[str]
    subsample_groups: list[str]
    backend: str
    notes: list[str] = field(default_factory=list)


class UnionFind:
    def __init__(self, keys) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key):
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, left, right, weights) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if weights.get(right_root, 0) > weights.get(left_root, 0):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        weights[left_root] = weights.get(left_root, 0) + weights.get(right_root, 0)
        return True


def mmseqs_binary() -> Optional[str]:
    env_bin = os.environ.get("MMSEQS_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin
    candidate = shutil.which("mmseqs")
    if candidate:
        return candidate
    search_paths = [
        Path(sys.prefix) / "bin" / "mmseqs",
        Path.home() / ".local" / "bin" / "mmseqs",
        Path("/opt/homebrew/bin/mmseqs"),
        Path("/usr/local/bin/mmseqs"),
        Path("/usr/bin/mmseqs"),
        Path("/opt/conda/bin/mmseqs"),
        Path("/opt/miniconda3/bin/mmseqs"),
        Path.home() / "miniconda3" / "bin" / "mmseqs",
        Path.home() / "anaconda3" / "bin" / "mmseqs",
    ]
    for p in search_paths:
        if p.is_file():
            return str(p)
    return None


def mmseqs_available() -> bool:
    return mmseqs_binary() is not None


def parasail_available() -> bool:
    try:
        import parasail
    except Exception:
        return False
    return parasail is not None


def biopython_available() -> bool:
    try:
        from Bio import Align
    except Exception:
        return False
    return Align is not None


def alignment_backend() -> str:
    if parasail_available():
        return "parasail"
    if biopython_available():
        return "biopython"
    return "none"


_ALIGNER: list = list()


def biopython_aligner():
    if not _ALIGNER:
        from Bio import Align
        from Bio.Align import substitution_matrices

        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -11.0
        aligner.extend_gap_score = -1.0
        _ALIGNER.append(aligner)
    return _ALIGNER[0]


def parasail_counts(query: str, target: str):
    import parasail

    result = parasail.nw_trace_scan_16(query, target, 11, 1, parasail.blosum62)
    trace = result.traceback
    matches = 0
    aligned = 0
    for left, right in zip(trace.query, trace.ref):
        if left == "-" or right == "-":
            continue
        aligned += 1
        if left == right:
            matches += 1
    return matches, aligned


def biopython_counts(query: str, target: str):
    alignment = biopython_aligner().align(query, target)[0]
    matches = 0
    aligned = 0
    for left_block, right_block in zip(alignment.aligned[0], alignment.aligned[1]):
        left = query[int(left_block[0]) : int(left_block[1])]
        right = target[int(right_block[0]) : int(right_block[1])]
        aligned += len(left)
        matches += sum(1 for first, second in zip(left, right) if first == second)
    return matches, aligned


def write_fasta(path: Path, sequences: list[str]) -> None:
    lines = []
    for position, sequence in enumerate(sequences):
        lines.append(f">s{position}")
        lines.append(sequence)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cluster_command(mmseqs_bin: str, db: Path, clu: Path, tmp: Path, min_seq_id: float, coverage: float, single_step) -> list[str]:
    command = [
        mmseqs_bin,
        "cluster",
        str(db),
        str(clu),
        str(tmp),
        "--min-seq-id",
        f"{min_seq_id}",
        "-c",
        f"{coverage}",
        "--cluster-mode",
        "1",
        "--max-seqs",
        "10000000",
        "-s",
        "7.5",
    ]
    if single_step:
        command.append("--single-step-clustering")
    return command


def run_mmseqs(sequences: list[str], min_seq_id: float, coverage: float, workdir: Path) -> dict[str, str]:
    binary = mmseqs_binary()
    if not binary:
        raise RuntimeError("mmseqs binary not found")
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    fasta = workdir / "input.fasta"
    write_fasta(fasta, sequences)
    db = workdir / "db"
    clu = workdir / "clu"
    tsv = workdir / "clu.tsv"
    tmp = workdir / "tmp"
    subprocess.run([binary, "createdb", str(fasta), str(db)], check=True, capture_output=True)
    failure = None
    for single_step in (False, True):
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            command = cluster_command(binary, db, clu, tmp, min_seq_id, coverage, single_step)
            subprocess.run(command, check=True, capture_output=True)
            failure = None
            break
        except subprocess.CalledProcessError as error:
            failure = error
    if failure is not None:
        raise RuntimeError(failure.stderr.decode("utf-8", "replace"))
    subprocess.run([binary, "createtsv", str(db), str(db), str(clu), str(tsv)], check=True, capture_output=True)
    mapping: dict[str, str] = {}
    def _parse_idx(v: str) -> int:
        return int(v[1:]) if v.startswith("s") else int(v)
    for line in tsv.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        representative = _parse_idx(fields[0])
        member = _parse_idx(fields[1])
        mapping[sequences[member]] = sequences[representative]
    for sequence in sequences:
        mapping.setdefault(sequence, sequence)
    return mapping


def alignment_similarity(query: str, target: str) -> float:
    backend = alignment_backend()
    if backend == "parasail":
        matches, aligned = parasail_counts(query, target)
    elif backend == "biopython":
        matches, aligned = biopython_counts(query, target)
    else:
        return 0.0
    if aligned == 0:
        return 0.0
    identity = matches / aligned
    span = aligned / max(len(query), len(target))
    return identity * span


def merge_by_alignment(labels: list[str], sequences: list[str], threshold: float, notes: list[str]) -> list[str]:
    backend = alignment_backend()
    if backend == "none":
        notes.append("no alignment backend available: cluster merge skipped, leakage audit incomplete")
        return list(labels)
    representatives: dict[str, str] = dict()
    weights: dict[str, int] = dict()
    for label, sequence in zip(labels, sequences):
        weights[label] = weights.get(label, 0) + 1
        current = representatives.get(label)
        if current is None or len(sequence) > len(current):
            representatives[label] = sequence
    keys = sorted(representatives)
    if len(keys) > MAX_ALIGNMENT_REPRESENTATIVES:
        notes.append(f"alignment merge skipped: {len(keys)} representatives exceed cap {MAX_ALIGNMENT_REPRESENTATIVES}")
        return list(labels)
    finder = UnionFind(keys)
    merged = 0
    for first in range(len(keys)):
        for second in range(first + 1, len(keys)):
            left = keys[first]
            right = keys[second]
            if finder.find(left) == finder.find(right):
                continue
            query = representatives[left]
            target = representatives[right]
            longer = max(len(query), len(target))
            if longer == 0 or min(len(query), len(target)) < threshold * longer:
                continue
            score = alignment_similarity(query, target)
            if score >= threshold and finder.union(left, right, weights):
                merged += 1
    notes.append(
        f"{backend} alignment merge collapsed {merged} cluster pairs at or above similarity {threshold} "
        f"across {len(keys)} representatives"
    )
    return [finder.find(label) for label in labels]


def canonical_labels(labels: list[str], prefix: str) -> list[str]:
    order: dict[str, str] = {}
    for label in labels:
        if label not in order:
            order[label] = f"{prefix}{len(order):05d}"
    return [order[label] for label in labels]


def antigen_labels(frame, min_seq_id, coverage, workdir, notes) -> list[str]:
    sequences = sorted(set(frame["Antigen_AA"]))
    mapping = run_mmseqs(sequences, float(min_seq_id), float(coverage), Path(workdir))
    notes.append(f"mmseqs2 clustered {len(sequences)} antigens at identity {float(min_seq_id):.2f}")
    return [mapping[sequence] for sequence in frame["Antigen_AA"]]


def antibody_labels(frame, min_seq_id, coverage, workdir, notes) -> list[str]:
    pool = list(frame["Antibody_VH_AA"]) + list(frame["Antibody_VL_AA"])
    chains = sorted({chain for chain in pool if chain})
    mapping = run_mmseqs(chains, float(min_seq_id), float(coverage), Path(workdir))
    notes.append(f"mmseqs2 clustered {len(chains)} antibody chains at identity {float(min_seq_id):.2f}")
    grouped = []
    for vh, vl in zip(frame["Antibody_VH_AA"], frame["Antibody_VL_AA"]):
        grouped.append([mapping[chain] for chain in (vh, vl) if chain])
    nodes = sorted({node for parts in grouped for node in parts})
    weights = {node: 1 for node in nodes}
    finder = UnionFind(nodes)
    for parts in grouped:
        for other in parts[1:]:
            finder.union(parts[0], other, weights)
    return [finder.find(parts[0]) for parts in grouped]


def fallback_antigen(frame, notes) -> list[str]:
    if "Cluster_id" in frame.columns and frame["Cluster_id"].notna().all() and frame["Cluster_id"].nunique() > 1:
        notes.append("mmseqs2 unavailable: reusing dataset Cluster_id as antigen clusters")
        return list(frame["Cluster_id"].astype(str))
    notes.append("mmseqs2 unavailable: using exact antigen sequence identity")
    return list(frame["Antigen_AA"].astype(str))


def fallback_antibody(frame, notes) -> list[str]:
    notes.append("mmseqs2 unavailable: using exact VH/VL sequence identity as antibody clusters")
    return [f"{vh}|{vl}" for vh, vl in zip(frame["Antibody_VH_AA"], frame["Antibody_VL_AA"])]


def build_clusters(
    frame: pd.DataFrame,
    config: Any = None,
    workdir: str | Path = "artifacts/cluster_tmp",
    cluster_on: str = "antigen",
    min_seq_id: float = 0.4,
    subsample_min_seq_id: float = 0.8,
    coverage: float = 0.4,
    parasail_merge: bool = True,
) -> ClusterAssignment:
    if config is not None:
        if hasattr(config, "cluster_on"):
            cluster_on = getattr(config, "cluster_on")
        elif isinstance(config, dict) and "cluster_on" in config:
            cluster_on = config["cluster_on"]

        if hasattr(config, "min_seq_id"):
            min_seq_id = float(getattr(config, "min_seq_id"))
        elif isinstance(config, dict) and "min_seq_id" in config:
            min_seq_id = float(config["min_seq_id"])

        if hasattr(config, "subsample_min_seq_id"):
            subsample_min_seq_id = float(getattr(config, "subsample_min_seq_id"))
        elif isinstance(config, dict) and "subsample_min_seq_id" in config:
            subsample_min_seq_id = float(config["subsample_min_seq_id"])

        if hasattr(config, "coverage"):
            coverage = float(getattr(config, "coverage"))
        elif isinstance(config, dict) and "coverage" in config:
            coverage = float(config["coverage"])

        if hasattr(config, "parasail_merge"):
            parasail_merge = bool(getattr(config, "parasail_merge"))
        elif isinstance(config, dict) and "parasail_merge" in config:
            parasail_merge = bool(config["parasail_merge"])

    notes: list[str] = []
    root = Path(workdir)
    backend = "mmseqs2" if mmseqs_available() else "identity"
    use_antibody = str(cluster_on).lower().startswith("antibod")
    same_threshold = abs(float(min_seq_id) - float(subsample_min_seq_id)) < 1e-9
    if backend == "mmseqs2":
        groups = antigen_labels(frame, subsample_min_seq_id, coverage, root / "subsample", notes)
        if use_antibody:
            splits = antibody_labels(frame, min_seq_id, coverage, root / "split", notes)
        elif same_threshold:
            splits = list(groups)
        else:
            splits = antigen_labels(frame, min_seq_id, coverage, root / "split", notes)
    else:
        groups = fallback_antigen(frame, notes)
        splits = fallback_antibody(frame, notes) if use_antibody else list(groups)
        if not use_antibody and not same_threshold:
            notes.append("identity backend cannot honour distinct split and subsample identity thresholds")
    if parasail_merge and parasail_available():
        if use_antibody:
            sequences = [f"{vh}{vl}" for vh, vl in zip(frame["Antibody_VH_AA"], frame["Antibody_VL_AA"])]
        else:
            sequences = list(frame["Antigen_AA"])
        splits = merge_by_alignment(splits, sequences, float(min_seq_id), notes)
    return ClusterAssignment(canonical_labels(splits, "s"), canonical_labels(groups, "g"), backend, notes)


def subsample(
    frame: pd.DataFrame,
    assignment: ClusterAssignment,
    max_per_cluster: int = 25,
    seed: int = 0
) -> Tuple[pd.DataFrame, ClusterAssignment]:
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = {}
    for position, group in enumerate(assignment.subsample_groups):
        buckets.setdefault(group, []).append(position)
    selected: list[int] = []
    for group in sorted(buckets):
        members = buckets[group]
        if max_per_cluster > 0 and len(members) > max_per_cluster:
            picked = rng.choice(np.asarray(members), size=int(max_per_cluster), replace=False)
            members = [int(value) for value in picked]
        selected.extend(int(value) for value in members)
    selected.sort()
    reduced = frame.iloc[selected].reset_index(drop=True)
    splits = [assignment.split_clusters[position] for position in selected]
    groups = [assignment.subsample_groups[position] for position in selected]
    notes = list(assignment.notes)
    notes.append(f"subsampled {len(frame)} pairs to {len(reduced)} with at most {max_per_cluster} per cluster")
    return reduced, ClusterAssignment(splits, groups, assignment.backend, notes)


def random_shuffle(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    count = len(frame)
    keys = list(frame["Antigen_AA"])
    permutation = rng.permutation(count)
    for _ in range(32):
        fixed = [position for position in range(count) if keys[permutation[position]] == keys[position]]
        if not fixed:
            break
        for position in fixed:
            other = int(rng.integers(count))
            permutation[position], permutation[other] = permutation[other], permutation[position]
    data = frame.copy()
    for column in ANTIGEN_BLOCK:
        if column in frame.columns:
            data[column] = frame[column].to_numpy()[permutation]
    if "Antibody_VH_AA" in data.columns and "Antibody_VL_AA" in data.columns and "Antigen_AA" in data.columns:
        rehashed = [pair_hash(row.Antibody_VH_AA, row.Antibody_VL_AA, row.Antigen_AA) for row in data.itertuples()]
        data["Unique_AG_VH_VL_Hash"] = rehashed
    return data.reset_index(drop=True)
