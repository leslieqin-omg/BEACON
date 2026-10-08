import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from beacon.config import BiEncoderConfig
from beacon.models.modules import DepthwiseSeparableConv1d, GatedAttentivePooling, ResidualGatedPooling


class SequenceTower(nn.Module):
    """
    General single-chain sequence encoder tower.
    Projects raw features -> passes through Depthwise-Separable Convs -> Pools into fixed vector.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_convs: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
        pooling_mode: str = "attention"
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(hidden_dim, kernel_size, dropout)
            for _ in range(num_convs)
        ])
        
        self.pooling_mode = pooling_mode
        if "residual" in pooling_mode or "sigmoid" in pooling_mode or "gate" in pooling_mode:
            self.pooler = ResidualGatedPooling(hidden_dim)
        elif "attention" in pooling_mode:
            self.pooler = GatedAttentivePooling(hidden_dim)
        else:
            self.pooler = None
            
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        region_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        features: [B, L, D_in]
        mask: [B, L] (padding mask, 1 for real residue, 0 for pad)
        region_weight: [B, L] (optional interface mask: CDR/Paratope for Ab, Epitope for Ag)
        Returns: normalized embedding [B, D_out]
        """
        h = self.input_proj(features)
        for conv in self.convs:
            h = conv(h, mask)
            
        # Optional: inject region weight (e.g. emphasize CDRs or Epitopes)
        if region_weight is not None:
            effective_mask = mask.bool() & (region_weight > 0) if mask is not None else (region_weight > 0)
            has_interface = effective_mask.any(dim=1, keepdim=True)
            base_m = mask.bool() if mask is not None else torch.ones_like(effective_mask)
            pool_mask = torch.where(has_interface, effective_mask, base_m)
        else:
            pool_mask = mask

        if self.pooler is not None:
            pooled = self.pooler(h, pool_mask)
        else:
            # Masked Mean Pooling
            if pool_mask is not None:
                weights = pool_mask.unsqueeze(-1).float()
                pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            else:
                pooled = h.mean(dim=1)

        embedding = self.out_proj(pooled)
        # L2 normalize for robust cosine similarity and vector indexing
        return F.normalize(embedding, p=2, dim=-1)


class BiEncoderScreening(nn.Module):
    """
    Stage 1: Asymmetric Bi-Encoder Screening Model.
    Designed for ultra-fast candidate retrieval and offline vector database indexing.
    Supports both Full-length (Ab-Ag) and Masked Interface (Pa-Ep) pooling modes.
    """
    def __init__(self, config: BiEncoderConfig):
        super().__init__()
        self.config = config
        
        # Antigen Tower (Global sequence representation or Epitope-focused)
        self.antigen_tower = SequenceTower(
            input_dim=config.antigen_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.projection_dim,
            num_convs=config.num_conv_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            pooling_mode=config.antigen_pooling
        )
        
        # Antibody Tower (CDR-focused or Full-length sequence representation)
        self.antibody_tower = SequenceTower(
            input_dim=config.antibody_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.projection_dim,
            num_convs=config.num_conv_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            pooling_mode=config.antibody_pooling
        )
        
        # Learnable inverse temperature parameter for contrastive alignment
        initial_logit_scale = math.log(1.0 / config.init_temperature)
        if config.learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * initial_logit_scale)
        else:
            self.register_buffer("logit_scale", torch.tensor(initial_logit_scale))

    def encode_antigen(
        self,
        antigen_feat: torch.Tensor,
        antigen_mask: Optional[torch.Tensor] = None,
        epi_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encodes antigen sequence into a compact vector for offline library caching."""
        return self.antigen_tower(antigen_feat, mask=antigen_mask, region_weight=epi_mask)

    def encode_antibody(
        self,
        antibody_feat: torch.Tensor,
        antibody_mask: Optional[torch.Tensor] = None,
        cdr_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encodes antibody sequence into an online query vector."""
        return self.antibody_tower(antibody_feat, mask=antibody_mask, region_weight=cdr_mask)

    def forward(
        self,
        antigen_feat: torch.Tensor,
        antibody_feat: torch.Tensor,
        antigen_mask: Optional[torch.Tensor] = None,
        antibody_mask: Optional[torch.Tensor] = None,
        cdr_mask: Optional[torch.Tensor] = None,
        epi_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates symmetric pairwise cosine similarity between paired antibody and antigen batches.
        """
        v_ag = self.encode_antigen(antigen_feat, antigen_mask, epi_mask=epi_mask)  # [B, D]
        u_ab = self.encode_antibody(antibody_feat, antibody_mask, cdr_mask=cdr_mask)  # [B, D]
        
        # Cosine similarity for true paired items
        paired_sim = (u_ab * v_ag).sum(dim=-1)  # [B]
        
        # Symmetric affinity matrix for in-batch contrastive learning
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        affinity_matrix = torch.matmul(u_ab, v_ag.t()) * logit_scale  # [B, B]
        
        return {
            "paired_similarity": paired_sim,
            "affinity_matrix": affinity_matrix,
            "antigen_embedding": v_ag,
            "antibody_embedding": u_ab,
            "logit_scale": logit_scale
        }
