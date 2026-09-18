from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_positive_matrix(labels: torch.Tensor) -> torch.Tensor:
    """Builds a binary mask indicating samples with identical biological identity."""
    return (labels.unsqueeze(1) == labels.unsqueeze(0)).to(torch.float32)


def multi_positive_infonce_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Multi-Positive InfoNCE loss.
    Avoids penalizing multiple occurrences of binding antibodies/antigens in the same batch.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    pos_counts = positive_mask.sum(dim=-1).clamp(min=1.0)
    loss = -(log_probs * positive_mask).sum(dim=-1) / pos_counts
    return loss.mean()


class MultiPositiveBiEncoderLoss(nn.Module):
    """
    Stage 1 Loss: Symmetric Multi-Positive InfoNCE + Optional Pair Binary Cross Entropy.
    Directly incorporates CALM's multi-positive label alignment strategy.
    """
    def __init__(self, use_multi_positive: bool = True):
        super().__init__()
        self.use_multi_positive = use_multi_positive

    def forward(
        self,
        affinity_matrix: torch.Tensor,
        antibody_labels: torch.Tensor = None,
        antigen_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        affinity_matrix: [B, B] scaled similarity logits
        antibody_labels: [B] integer class ID for unique antibody sequence
        antigen_labels: [B] integer class ID for unique antigen sequence
        """
        B = affinity_matrix.size(0)
        device = affinity_matrix.device

        if self.use_multi_positive and antibody_labels is not None and antigen_labels is not None:
            # Multi-positive target matrices
            fwd_pos = build_positive_matrix(antigen_labels)
            rev_pos = build_positive_matrix(antibody_labels)
            
            loss_ab2ag = multi_positive_infonce_loss(affinity_matrix, fwd_pos)
            loss_ag2ab = multi_positive_infonce_loss(affinity_matrix.t(), rev_pos)
        else:
            # Fallback to standard 1-to-1 diagonal targets
            targets = torch.arange(B, device=device)
            loss_ab2ag = F.cross_entropy(affinity_matrix, targets)
            loss_ag2ab = F.cross_entropy(affinity_matrix.t(), targets)

        return 0.5 * (loss_ab2ag + loss_ag2ab)

