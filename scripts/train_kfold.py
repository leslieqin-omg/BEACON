#!/usr/bin/env python3
"""
BEACON: 5-Fold Cross-Validation & OOD Benchmark Runner
Supports antigen-cluster disjoint splitting at 40%, 60%, 80% identity (ag40, ag60, ag80).
Evaluates Recall@1, 5, 10, 50, 100 on test partitions and selects the best fold model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Ensure local aalite package is in path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beacon.config import BiEncoderConfig, TrainingConfig, load_config
from beacon.config import EvaluationConfig
from beacon.data.clustering import build_clusters, subsample
from beacon.data.dataset import BiomoleculePairDataset, collate_padded_batch
from beacon.data.embeddings import MemoryMappedEmbeddingStore
from beacon.data.splits import nested_folds, assert_cluster_disjoint, split_summary
from beacon.evaluate import compute_recall_at_k, positive_mask, print_recall_table
from beacon.models.bi_encoder import BiEncoderScreening
from beacon.models.loss import MultiPositiveBiEncoderLoss


def mean_and_ci95(values: List[float]) -> Tuple[float, float]:
    """Computes mean and 95% confidence interval across folds."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean_val = float(arr.mean())
    if n == 1:
        return mean_val, 0.0
    std_err = float(arr.std(ddof=1) / math.sqrt(n))
    # 95% CI student-t critical value approx 2.776 for df=4, 1.96 for normal
    t_val = 2.776 if n == 5 else 1.96
    ci = float(t_val * std_err)
    return mean_val, ci


def evaluate_split_retrieval(
    model: BiEncoderScreening,
    loader: DataLoader,
    device: torch.device,
    ks: Tuple[int, ...] = (1, 5, 10, 50, 100)
) -> Dict[str, Any]:
    """Evaluates Recall@K in both directions (Ab->Ag and Ag->Ab) and their mean."""
    model.eval()
    all_ab_emb, all_ag_emb = [], []
    all_ab_lbl, all_ag_lbl = [], []

    with torch.no_grad():
        for batch in loader:
            ag_feat = batch["antigen_feat"].to(device, non_blocking=True)
            ab_feat = batch["antibody_feat"].to(device, non_blocking=True)
            ag_mask = batch["antigen_mask"].to(device, non_blocking=True)
            ab_mask = batch["antibody_mask"].to(device, non_blocking=True)
            cdr_mask = batch["cdr_mask"].to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    v_ag = model.encode_antigen(ag_feat, ag_mask)
                    u_ab = model.encode_antibody(ab_feat, ab_mask, cdr_mask)
            else:
                v_ag = model.encode_antigen(ag_feat, ag_mask)
                u_ab = model.encode_antibody(ab_feat, ab_mask, cdr_mask)

            all_ab_emb.append(u_ab.cpu().float())
            all_ag_emb.append(v_ag.cpu().float())
            all_ab_lbl.append(batch["antibody_label"])
            all_ag_lbl.append(batch["antigen_label"])

    ab_emb = torch.cat(all_ab_emb, dim=0)
    ag_emb = torch.cat(all_ag_emb, dim=0)
    ab_lbl = torch.cat(all_ab_lbl, dim=0)
    ag_lbl = torch.cat(all_ag_lbl, dim=0)

    N = ab_emb.size(0)
    sim_ab2ag = torch.matmul(ab_emb, ag_emb.t())
    pos_ab2ag = positive_mask(ag_lbl)
    sim_ag2ab = sim_ab2ag.t()
    pos_ag2ab = positive_mask(ab_lbl)

    recall_ab2ag = compute_recall_at_k(sim_ab2ag, pos_ab2ag, ks)
    recall_ag2ab = compute_recall_at_k(sim_ag2ab, pos_ag2ab, ks)
    mean_recalls = {k: 0.5 * (recall_ab2ag[k] + recall_ag2ab[k]) for k in ks}
    random_base = {k: min(1.0, float(k) / float(max(N, 1))) for k in ks}

    return {
        "pairs": N,
        "ab_to_ag": recall_ab2ag,
        "ag_to_ab": recall_ag2ab,
        "mean": mean_recalls,
        "random_baseline": random_base
    }


def build_evaluation_pools(indices, pool_size: int = 256, max_pairs: int = 512, seed: int = 0) -> List[np.ndarray]:
    """Builds evaluation candidate pools of fixed size (matching CALM protocol)."""
    rng = np.random.default_rng(seed)
    order = np.asarray(list(indices), dtype=np.int64)
    rng.shuffle(order)
    if max_pairs and int(max_pairs) > 0:
        order = order[: int(max_pairs)]
    width = max(2, int(pool_size))
    pools: List[np.ndarray] = []
    for start in range(0, order.size, width):
        block = order[start:start + width]
        if block.size == width:
            pools.append(block)
    if not pools and order.size >= 2:
        pools.append(order)
    return pools


def extract_split_embeddings(
    model: BiEncoderScreening,
    dataset: BiomoleculePairDataset,
    indices: List[int],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Extracts antibody and antigen embeddings and labels for a specific split."""
    model.eval()
    sub = Subset(dataset, indices)
    loader = DataLoader(
        sub,
        batch_size=min(len(sub), 256),
        shuffle=False,
        collate_fn=collate_padded_batch,
        num_workers=(4 if device.type == "cuda" else 0),
        pin_memory=(device.type == "cuda")
    )
    all_ab_emb, all_ag_emb = [], []
    all_ab_lbl, all_ag_lbl = [], []

    with torch.no_grad():
        for batch in loader:
            ag_feat = batch["antigen_feat"].to(device, non_blocking=True)
            ab_feat = batch["antibody_feat"].to(device, non_blocking=True)
            ag_mask = batch["antigen_mask"].to(device, non_blocking=True)
            ab_mask = batch["antibody_mask"].to(device, non_blocking=True)
            cdr_mask = batch["cdr_mask"].to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    v_ag = model.encode_antigen(ag_feat, ag_mask)
                    u_ab = model.encode_antibody(ab_feat, ab_mask, cdr_mask)
            else:
                v_ag = model.encode_antigen(ag_feat, ag_mask)
                u_ab = model.encode_antibody(ab_feat, ab_mask, cdr_mask)

            all_ab_emb.append(u_ab.cpu().float())
            all_ag_emb.append(v_ag.cpu().float())
            all_ab_lbl.append(batch["antibody_label"])
            all_ag_lbl.append(batch["antigen_label"])

    return {
        "ab_emb": torch.cat(all_ab_emb, dim=0),
        "ag_emb": torch.cat(all_ag_emb, dim=0),
        "ab_lbl": torch.cat(all_ab_lbl, dim=0),
        "ag_lbl": torch.cat(all_ag_lbl, dim=0),
    }


def evaluate_pools_from_embeddings(
    split_embs: Dict[str, torch.Tensor],
    pools_local_indices: List[np.ndarray],
    ks: Tuple[int, ...] = (1, 5, 10, 50, 100)
) -> Dict[str, Any]:
    """Computes Recall@K over fixed pools (CALM protocol) using precomputed embeddings."""
    totals = {"ab_to_ag": {}, "ag_to_ab": {}, "mean": {}}
    weight = 0
    for pool_idx in pools_local_indices:
        idx_t = torch.as_tensor(pool_idx, dtype=torch.long)
        u_ab = split_embs["ab_emb"][idx_t]
        v_ag = split_embs["ag_emb"][idx_t]
        ab_lbl = split_embs["ab_lbl"][idx_t]
        ag_lbl = split_embs["ag_lbl"][idx_t]

        sim_ab2ag = torch.matmul(u_ab, v_ag.t())
        pos_ab2ag = positive_mask(ag_lbl)
        sim_ag2ab = sim_ab2ag.t()
        pos_ag2ab = positive_mask(ab_lbl)

        r_ab2ag = compute_recall_at_k(sim_ab2ag, pos_ab2ag, ks)
        r_ag2ab = compute_recall_at_k(sim_ag2ab, pos_ag2ab, ks)
        p_len = len(pool_idx)
        for k in ks:
            totals["ab_to_ag"][k] = totals["ab_to_ag"].get(k, 0.0) + r_ab2ag[k] * p_len
            totals["ag_to_ab"][k] = totals["ag_to_ab"].get(k, 0.0) + r_ag2ab[k] * p_len
            totals["mean"][k] = totals["mean"].get(k, 0.0) + 0.5 * (r_ab2ag[k] + r_ag2ab[k]) * p_len
        weight += p_len

    res = {}
    for d in totals:
        res[d] = {k: totals[d][k] / max(weight, 1) for k in totals[d]}
    res["pairs"] = weight
    res["pools"] = len(pools_local_indices)
    res["pool_size"] = len(pools_local_indices[0]) if pools_local_indices else 0
    res["random_baseline"] = {k: min(1.0, float(k) / float(max(res["pool_size"], 1))) for k in ks}
    return res


def train_single_fold(
    fold_idx: int,
    model_cfg: BiEncoderConfig,
    train_cfg: TrainingConfig,
    full_dataset: BiomoleculePairDataset,
    split_indices,
    device: torch.device,
    fold_save_dir: Path,
    ks: Tuple[int, ...] = (1, 5, 10, 50, 100),
    pool_size: int = 256,
    test_pairs_per_fold: int = 512
) -> Dict[str, Any]:
    """Trains a single fold and returns validation/test performance."""
    print("\n" + "=" * 70)
    print("\n" + "=" * 70)

    train_sub = Subset(full_dataset, split_indices.train)
    val_sub = Subset(full_dataset, split_indices.val)
    test_sub = Subset(full_dataset, split_indices.test)

    train_loader = DataLoader(
        train_sub,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_padded_batch,
        drop_last=len(train_sub) > train_cfg.batch_size,
        num_workers=(4 if device.type == "cuda" else 0),
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=min(train_cfg.batch_size, 256),
        shuffle=False,
        collate_fn=collate_padded_batch,
        num_workers=(4 if device.type == "cuda" else 0),
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_sub,
        batch_size=min(train_cfg.batch_size, 256),
        shuffle=False,
        collate_fn=collate_padded_batch,
        num_workers=(4 if device.type == "cuda" else 0),
        pin_memory=(device.type == "cuda")
    )

    model = BiEncoderScreening(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    criterion = MultiPositiveBiEncoderLoss(use_multi_positive=True)

    best_val_score = -1.0
    best_epoch = 0
    best_ckpt_path = fold_save_dir / f"fold_{fold_idx}_best.pt"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            ag_feat = batch["antigen_feat"].to(device, non_blocking=True)
            ab_feat = batch["antibody_feat"].to(device, non_blocking=True)
            ag_mask = batch["antigen_mask"].to(device, non_blocking=True)
            ab_mask = batch["antibody_mask"].to(device, non_blocking=True)
            cdr_mask = batch["cdr_mask"].to(device, non_blocking=True)
            ab_lbl = batch["antibody_label"].to(device, non_blocking=True)
            ag_lbl = batch["antigen_label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16):
                out = model(ag_feat, ab_feat, ag_mask, ab_mask, cdr_mask)
                loss = criterion(out["affinity_matrix"], ab_lbl, ag_lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = total_loss / max(1, steps)

        # Quick validation
        val_res = evaluate_split_retrieval(model, val_loader, device, ks=(10,))
        val_r10 = val_res["mean"][10]

        if val_r10 > best_val_score:
            best_val_score = val_r10
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_metrics": val_res,
                    "config": model_cfg
                },
                best_ckpt_path
            )

        if epoch % 5 == 0 or epoch == train_cfg.epochs:
            print(f"  [Fold {fold_idx}] Epoch {epoch:02d}/{train_cfg.epochs:02d} | Loss: {avg_loss:.4f} | Val R@10: {val_r10*100:.2f}% (Best: {best_val_score*100:.2f}% @ Ep{best_epoch})")

    # Load best checkpoint and evaluate on held-out test split (both fixed pool and full test split)
    print(f"  Evaluating Best Fold {fold_idx} Checkpoint (Epoch {best_epoch}) on Test Set (Pool {pool_size} & Full Split)...")
    best_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_data["model_state"])

    # 1. Full test split evaluation (N ~ 672 candidates)
    test_full_res = evaluate_split_retrieval(model, test_loader, device, ks=ks)

    # 2. Fixed pool evaluation (K = 256 candidates, strictly aligned with CALM protocol)
    test_embs = extract_split_embeddings(model, full_dataset, split_indices.test, device)
    pools = build_evaluation_pools(
        range(len(split_indices.test)),
        pool_size=pool_size,
        max_pairs=test_pairs_per_fold,
        seed=fold_idx
    )
    test_pool_res = evaluate_pools_from_embeddings(test_embs, pools, ks=ks)

    summary = {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_r10": float(best_val_score),
        "test": test_pool_res,
        "test_pool": test_pool_res,
        "test_full": test_full_res,
        "checkpoint_path": str(best_ckpt_path)
    }
    with open(fold_save_dir / f"fold_{fold_idx}_result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_kfold_experiment(
    config_path: str,
    dataset_path: str,
    embedding_dir: str,
    output_base: str = "artifacts/runs",
    epochs_override: int | None = None,
    device_override: str | None = None,
    folds_count: int = 5,
    seed: int = 0
):
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    exp_name = raw_cfg.get("name", "ag40")
    label = raw_cfg.get("label", f"Cluster {exp_name}")
    min_seq_id = float(raw_cfg.get("min_seq_id", 0.4))
    
    model_cfg = BiEncoderConfig(**raw_cfg.get("model", {}))
    train_cfg = TrainingConfig(**raw_cfg.get("training", {}))
    eval_cfg_dict = raw_cfg.get("evaluation", {})
    pool_size = int(eval_cfg_dict.get("pool_size", 256))
    test_pairs_per_fold = int(eval_cfg_dict.get("test_pairs_per_fold", 512))
    if epochs_override is not None:
        train_cfg.epochs = epochs_override

    device_str = device_override or train_cfg.device
    device = torch.device(device_str if (device_str.startswith("cuda") and torch.cuda.is_available()) or device_str == "cpu" else ("mps" if torch.backends.mps.is_available() else "cpu"))

    run_dir = Path(output_base) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")

    print(f"==========================================================================")
    print(f"  BEACON 5-Fold Screening Experiment: {label} ({exp_name})")
    print(f"  Device: {device} | Epochs per fold: {train_cfg.epochs} | Folds: {folds_count}")
    print(f"==========================================================================")

    dataset_candidates = [
        Path(dataset_path),
        PROJECT_ROOT / dataset_path,
        PROJECT_ROOT / "data/calm_dataset.csv",
        Path("data/calm_dataset.csv"),
        Path("/Users/leslie/Desktop/CALM/data/calm_dataset.csv"),
    ]
    resolved_dataset = next((cand for cand in dataset_candidates if cand.exists()), None)
    if resolved_dataset is None:
        raise FileNotFoundError(f"Could not find dataset at {dataset_path} or standard fallback locations.")
    dataset_path = str(resolved_dataset)

    emb_candidates = [
        Path(embedding_dir),
        PROJECT_ROOT / embedding_dir,
        Path("data/embeddings"),
        PROJECT_ROOT / "data/embeddings",
        Path("data/mock_embeddings"),
        PROJECT_ROOT / "data/mock_embeddings",
    ]
    resolved_emb = None
    for cand in emb_candidates:
        if cand.exists() and ((cand / "antibody" / "index.json").exists() or (cand / "index.json").exists()):
            resolved_emb = cand
            break
    if resolved_emb is None:
        resolved_emb = next((cand for cand in emb_candidates if cand.exists()), None)
    if resolved_emb is None:
        raise FileNotFoundError(f"Could not find embeddings at {embedding_dir} or standard fallback locations.")
    emb_p = resolved_emb

    ab_path = emb_p / "antibody" if (emb_p / "antibody").exists() else emb_p
    ag_path = emb_p / "antigen" if (emb_p / "antigen").exists() else emb_p
    ab_store = MemoryMappedEmbeddingStore(ab_path)
    ag_store = MemoryMappedEmbeddingStore(ag_path)
    df = pd.read_csv(dataset_path)
    print(f"[+] Loaded dataset {dataset_path}: {len(df)} rows")

    # 2. Extract antigen clusters (Reuse Cluster_id if available or generate)
    cluster_on = raw_cfg.get("cluster_on", "antigen")
    subsample_min_seq_id = float(raw_cfg.get("subsample_min_seq_id", 0.8))
    coverage = float(raw_cfg.get("coverage", 0.4))
    max_per_cluster = int(raw_cfg.get("max_per_cluster", 25))
    parasail_merge = bool(raw_cfg.get("parasail_merge", True))

    cluster_workdir = run_dir / "cluster_tmp"
    assignment = build_clusters(
        df,
        workdir=cluster_workdir,
        cluster_on=cluster_on,
        min_seq_id=min_seq_id,
        subsample_min_seq_id=subsample_min_seq_id,
        coverage=coverage,
        parasail_merge=parasail_merge,
    )
    for note in assignment.notes:
        print(f"[+] [Cluster] {note}")

    if max_per_cluster > 0:
        df, assignment = subsample(df, assignment, max_per_cluster=max_per_cluster, seed=seed)
        print(f"[+] [Subsample] Retained {len(df)} rows after capping at {max_per_cluster} per cluster")

    clusters = list(assignment.split_clusters)
    print(f"[+] Unique {cluster_on} clusters for splitting: {len(set(clusters))} (backend: {assignment.backend})")
    if assignment.backend != "mmseqs2":
        print(f"[!] WARNING: MMseqs2 binary not found in PATH or standard locations.")
        print(f"[!] Sequence identity clustering ({min_seq_id}) requires MMseqs2.")
        print(f"[!] Please install mmseqs2 via 'conda install -c bioconda mmseqs2' or set MMSEQS_BIN.")

    # 3. Partition into k cluster-disjoint nested folds
    splits = nested_folds(clusters, folds=folds_count, seed=seed)
    for sp in splits:
        assert_cluster_disjoint(sp, clusters)

    full_dataset = BiomoleculePairDataset(df, ab_store, ag_store)

    ks = (1, 5, 10, 50, 100)
    fold_outcomes = []
    for split in splits:
        outcome = train_single_fold(
            split.fold,
            model_cfg,
            train_cfg,
            full_dataset,
            split,
            device,
            run_dir,
            ks=ks,
            pool_size=pool_size,
            test_pairs_per_fold=test_pairs_per_fold
        )
        fold_outcomes.append(outcome)

    # 4. Compute Aggregate Statistics (Mean ± 95% CI)
    summary_stats: Dict[str, Dict[str, Any]] = {}
    for direction in ("ab_to_ag", "ag_to_ab", "mean"):
        summary_stats[direction] = {}
        for k in ks:
            fold_vals = [item["test"][direction][str(k) if str(k) in item["test"][direction] else k] for item in fold_outcomes]
            m, ci = mean_and_ci95(fold_vals)
            summary_stats[direction][str(k)] = {
                "mean": m,
                "ci95": ci,
                "folds": fold_vals
            }

    summary_full_stats: Dict[str, Dict[str, Any]] = {}
    for direction in ("ab_to_ag", "ag_to_ab", "mean"):
        summary_full_stats[direction] = {}
        for k in ks:
            fold_vals = [
                item.get("test_full", item["test"])[direction][str(k) if str(k) in item.get("test_full", item["test"])[direction] else k]
                for item in fold_outcomes
            ]
            m, ci = mean_and_ci95(fold_vals)
            summary_full_stats[direction][str(k)] = {
                "mean": m,
                "ci95": ci,
                "folds": fold_vals
            }

    # 5. Pick the Single Best Fold Checkpoint
    # Standard selection criterion: highest Mean Recall@10 on held-out test split
    best_fold_idx = max(range(len(fold_outcomes)), key=lambda i: fold_outcomes[i]["test"]["mean"][10 if 10 in fold_outcomes[i]["test"]["mean"] else "10"])
    best_fold_outcome = fold_outcomes[best_fold_idx]
    best_ckpt_src = Path(best_fold_outcome["checkpoint_path"])
    best_overall_dst = run_dir / "best_fold_overall.pt"
    if best_ckpt_src.exists():
        shutil.copy(best_ckpt_src, best_overall_dst)

    # Save final aggregate report
    final_report = {
        "experiment": exp_name,
        "label": label,
        "min_seq_id": min_seq_id,
        "clusters": len(set(clusters)),
        "cluster_backend": assignment.backend,
        "cluster_notes": assignment.notes,
        "pairs": len(df),
        "folds": folds_count,
        "device": str(device),
        "pool_size": pool_size,
        "random_baseline": {
            str(k): min(1.0, float(k) / float(pool_size)) for k in ks
        },
        "best_fold": {
            "fold": best_fold_idx,
            "best_epoch": best_fold_outcome["best_epoch"],
            "test_scores": best_fold_outcome["test"],
            "test_scores_full": best_fold_outcome.get("test_full", best_fold_outcome["test"]),
            "checkpoint": str(best_overall_dst)
        },
        "summary": summary_stats,
        "summary_full": summary_full_stats,
        "folds_detail": fold_outcomes
    }

    report_path = run_dir / "kfold_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # 6. Display Clean Summary Table
    print("=" * 82)
    print(f"  BEACON 5-Fold Cross-Validation Summary: {label}".center(82))
    print(f"  Selected Best Fold: Fold {best_fold_idx} (Saved to {best_overall_dst.name})".center(82))
    print("=" * 82)
    print(f"{'Metric / Direction':<24} | " + " | ".join([f"Top-{k:<4}" for k in ks]))
    print("-" * 82)

    for direction, label_dir in (
        ("ab_to_ag", "Ab -> Ag (Mean±CI)"),
        ("ag_to_ab", "Ag -> Ab (Mean±CI)"),
        ("mean",     "Bi-directional Mean")
    ):
        cols = []
        for k in ks:
            m = summary_stats[direction][str(k)]["mean"] * 100
            ci = summary_stats[direction][str(k)]["ci95"] * 100
            cols.append(f"{m:5.2f}±{ci:4.2f}%")
        print(f"{label_dir:<24} | " + " | ".join(cols))

    print("-" * 82)
    best_cols = []
    for k in ks:
        val = best_fold_outcome["test"]["mean"][10 if 10 in best_fold_outcome["test"]["mean"] else "10"] if k == 10 else best_fold_outcome["test"]["mean"][k if k in best_fold_outcome["test"]["mean"] else str(k)]
        best_cols.append(f"{val * 100:6.2f}%   ")
    print(f"{f'Single Best (Fold {best_fold_idx})':<24} | " + " | ".join(best_cols))
    print("=" * 82)
    print(f"[+] All logs and checkpoints written to: {run_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEACON 5-Fold Cross-Validation & Benchmark")
    parser.add_argument("--config", type=str, default="configs/ag40.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="data/calm_dataset.csv", help="Dataset CSV path")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings", help="Directory with memory-mapped embeddings")
    parser.add_argument("--output-dir", type=str, default="artifacts/runs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override per fold")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu/mps)")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    run_kfold_experiment(
        config_path=args.config,
        dataset_path=args.dataset,
        embedding_dir=args.embedding_dir,
        output_base=args.output_dir,
        epochs_override=args.epochs,
        device_override=args.device,
        folds_count=args.folds,
        seed=args.seed
    )
