import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DepthwiseSeparableConv1d(nn.Module):
    """
    Lightweight 1D Depthwise-Separable Convolution to capture local sequence motifs
    with minimal parameter footprint.
    """
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, D]
        residual = x
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        # Conv1d expects [B, D, L]
        h = x.transpose(1, 2)
        h = self.depthwise(h)
        h = self.pointwise(h)
        h = h.transpose(1, 2)
        h = self.norm(self.act(h))
        h = self.dropout(h)
        out = residual + h
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class GatedAttentivePooling(nn.Module):
    """
    Learned attention-based pooling layer to collapse variable-length sequence representations
    into a single fixed-dimensional embedding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, D]
        mask: [B, L] (1 for valid residue, 0 for padding)
        Returns: [B, D]
        """
        scores = self.scorer(x).squeeze(-1)  # [B, L]
        if mask is not None:
            fill_val = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(~mask.bool(), fill_val)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        pooled = (x * weights).sum(dim=1)  # [B, D]
        return pooled

