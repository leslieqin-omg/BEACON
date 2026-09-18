from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from beacon.config import BiEncoderConfig, TrainingConfig
from beacon.models.bi_encoder import BiEncoderScreening
from beacon.models.loss import MultiPositiveBiEncoderLoss


def train_epoch(
    model: BiEncoderScreening,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad(set_to_none=True)
        
        ag_feat = batch["antigen_feat"].to(device, non_blocking=True)
        ab_feat = batch["antibody_feat"].to(device, non_blocking=True)
        ag_mask = batch["antigen_mask"].to(device, non_blocking=True)
        ab_mask = batch["antibody_mask"].to(device, non_blocking=True)
        cdr_mask = batch["cdr_mask"].to(device, non_blocking=True)
        
        ab_lbl = batch["antibody_label"].to(device, non_blocking=True)
        ag_lbl = batch["antigen_label"].to(device, non_blocking=True)
        
        # Mixed Precision forward pass
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            out = model(
                antigen_feat=ag_feat,
                antibody_feat=ab_feat,
                antigen_mask=ag_mask,
                antibody_mask=ab_mask,
                cdr_mask=cdr_mask
            )
            loss = criterion(out["affinity_matrix"], ab_lbl, ag_lbl)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        steps += 1
        
    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate_retrieval(
    model: BiEncoderScreening,
    loader: DataLoader,
    device: torch.device,
    top_ks: tuple[int, ...] = (1, 5, 10, 50)
) -> Dict[str, float]:
    model.eval()
    all_ab_emb = []
    all_ag_emb = []
    all_ab_lbl = []
    all_ag_lbl = []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        ag_feat = batch["antigen_feat"].to(device, non_blocking=True)
        ab_feat = batch["antibody_feat"].to(device, non_blocking=True)
        ag_mask = batch["antigen_mask"].to(device, non_blocking=True)
        ab_mask = batch["antibody_mask"].to(device, non_blocking=True)
        cdr_mask = batch["cdr_mask"].to(device, non_blocking=True)
        
        v_ag = model.encode_antigen(ag_feat, ag_mask)
        u_ab = model.encode_antibody(ab_feat, ab_mask, cdr_mask)
        
        all_ag_emb.append(v_ag.cpu())
        all_ab_emb.append(u_ab.cpu())
        all_ab_lbl.append(batch["antibody_label"])
        all_ag_lbl.append(batch["antigen_label"])
        
    all_ab_emb = torch.cat(all_ab_emb, dim=0)  # [N, D]
    all_ag_emb = torch.cat(all_ag_emb, dim=0)  # [N, D]
    all_ab_lbl = torch.cat(all_ab_lbl, dim=0)  # [N]
    all_ag_lbl = torch.cat(all_ag_lbl, dim=0)  # [N]
    
    # Cosine Similarity Matrix over full evaluation set
    sim_matrix = torch.matmul(all_ab_emb, all_ag_emb.t())  # [N, N]
    
    # Ground truth: positive if antigen labels match
    pos_mask = (all_ag_lbl.unsqueeze(0) == all_ag_lbl.unsqueeze(1))  # [N, N]
    
    results = {}
    N = sim_matrix.size(0)
    for k in top_ks:
        if k > N:
            continue
        top_k_indices = torch.topk(sim_matrix, k=k, dim=1).indices
        hits = pos_mask.gather(1, top_k_indices).any(dim=1).float()
        results[f"Recall@{k}"] = hits.mean().item()
        
    return results


def train_screening_model(
    model_cfg: BiEncoderConfig,
    train_cfg: TrainingConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    checkpoint_dir: str = "artifacts/checkpoints"
) -> BiEncoderScreening:
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing training on device: {device}")
    
    model = BiEncoderScreening(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )
    
    # Cosine Annealing with Warm Restarts (same as CALM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    criterion = MultiPositiveBiEncoderLoss(use_multi_positive=True)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_score = -1.0
    
    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, train_cfg.grad_clip_norm
        )
        scheduler.step()
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch:02d}/{train_cfg.epochs:02d} | Train Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")
        
        if val_loader is not None:
            metrics = evaluate_retrieval(model, val_loader, device)
            metric_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            print(f"       Validation | {metric_str}")
            
            # Save checkpoint on top Recall@10
            r10 = metrics.get("Recall@10", 0.0)
            if r10 > best_score:
                best_score = r10
                ckpt_path = os.path.join(checkpoint_dir, "best_biencoder.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, ckpt_path)
                print(f"       [*] New best Recall@10 ({r10:.3f}) saved to {ckpt_path}")
                
    return model

