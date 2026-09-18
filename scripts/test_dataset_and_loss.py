import torch
import pandas as pd
from beacon.models.loss import MultiPositiveBiEncoderLoss
from beacon.models.bi_encoder import BiEncoderScreening
from beacon.config import BiEncoderConfig

def run_test():
    print("=== Testing CALM-style Multi-Positive Loss & Model Compatibility ===")
    cfg = BiEncoderConfig(antigen_dim=1280, antibody_dim=512, hidden_dim=256, projection_dim=128)
    model = BiEncoderScreening(cfg)
    criterion = MultiPositiveBiEncoderLoss(use_multi_positive=True)

    B = 8
    L_ag = 120
    L_ab = 80

    ag_feat = torch.randn(B, L_ag, 1280)
    ab_feat = torch.randn(B, L_ab, 512)
    ag_mask = torch.ones(B, L_ag)
    ab_mask = torch.ones(B, L_ab)
    cdr_mask = torch.randint(0, 2, (B, L_ab)).float()

    # Simulate multi-positive targets (e.g. sample 0 and 2 share the same antigen)
    ab_lbl = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    ag_lbl = torch.tensor([0, 1, 0, 2, 3, 4, 3, 5])

    out = model(ag_feat, ab_feat, ag_mask, ab_mask, cdr_mask)
    loss = criterion(out["affinity_matrix"], ab_lbl, ag_lbl)

    print(f"Affinity matrix shape: {out['affinity_matrix'].shape}")
    print(f"Calculated Multi-Positive InfoNCE Loss: {loss.item():.4f}")
    loss.backward()
    print("Gradient backward test passed successfully!")
    print("=== All CALM-aligned Stage 1 Modules Ready! ===")

if __name__ == "__main__":
    run_test()

