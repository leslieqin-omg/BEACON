from __future__ import annotations

from typing import Dict, Sequence, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def positive_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels.unsqueeze(1) == labels.unsqueeze(0)


def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    positives: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10, 50, 100)
) -> Dict[int, float]:
    num_queries, num_candidates = similarity_matrix.shape
    max_k = min(max(ks), num_candidates)
    ranked_indices = similarity_matrix.topk(max_k, dim=1).indices
    hits = torch.gather(positives, 1, ranked_indices)
    
    recalls = {}
    for k in ks:
        limit = min(k, max_k)
        hit_at_k = hits[:, :limit].any(dim=1).float().mean().item()
        recalls[int(k)] = float(hit_at_k)
    return recalls


def evaluate_retrieval_benchmarks(
    model,
    dataloader,
    device: torch.device,
    ks: tuple[int, ...] = (1, 5, 10, 50, 100)
) -> Dict[str, Any]:
    model.eval()
    all_ab_emb = []
    all_ag_emb = []
    all_ab_lbl = []
    all_ag_lbl = []
    
    print(f"[+] Encoding evaluation pairs on {device}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
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
    print(f"[+] Total evaluated pairs: {N}")
    
    sim_ab2ag = torch.matmul(ab_emb, ag_emb.t())
    pos_ab2ag = positive_mask(ag_lbl)
    
    sim_ag2ab = sim_ab2ag.t()
    pos_ag2ab = positive_mask(ab_lbl)
    
    recall_ab2ag = compute_recall_at_k(sim_ab2ag, pos_ab2ag, ks)
    recall_ag2ab = compute_recall_at_k(sim_ag2ab, pos_ag2ab, ks)
    
    mean_recalls = {k: 0.5 * (recall_ab2ag[k] + recall_ag2ab[k]) for k in ks}
    random_base = {k: min(1.0, float(k) / float(N)) for k in ks}
    
    return {
        "num_pairs": N,
        "ab_to_ag": recall_ab2ag,
        "ag_to_ab": recall_ag2ab,
        "mean": mean_recalls,
        "random_baseline": random_base,
    }


def print_recall_table(results: Dict[str, Any], title: str = "BEACON Stage 1 Screening Evaluation"):
    ks = sorted(list(results["mean"].keys()))
    border = "=" * 76
    sep = "-" * 76
    print()
    print(border)
    print(title.center(76))
    print(f"Candidates: {results['num_pairs']} | Metrics: Recall@K (%)".center(76))
    print(border)
    header = f"{'Direction':<22} | " + " | ".join([f"Top-{k:<4}" for k in ks])
    print(header)
    print(sep)
    
    row_ab = [f"{results['ab_to_ag'][k] * 100:6.2f}%" for k in ks]
    print(f"{'Antibody -> Antigen':<22} | " + " | ".join(row_ab))
    
    row_ag = [f"{results['ag_to_ab'][k] * 100:6.2f}%" for k in ks]
    print(f"{'Antigen -> Antibody':<22} | " + " | ".join(row_ag))
    
    row_mean = [f"{results['mean'][k] * 100:6.2f}%" for k in ks]
    print(f"{'Bi-directional Mean':<22} | " + " | ".join(row_mean))
    
    row_rand = [f"{results['random_baseline'][k] * 100:6.2f}%" for k in ks]
    print(f"{'Random Baseline':<22} | " + " | ".join(row_rand))
    print(border)
    print()
