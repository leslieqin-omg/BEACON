import sys
import torch

from beacon.config import load_config
from beacon.models import BiEncoderScreening, BiEncoderLoss

def test_bi_encoder():
    print("=== Testing BEACON Stage 1: BiEncoderScreening ===")
    config = load_config("configs/stage1_biencoder.yaml")
    model = BiEncoderScreening(config)
    criterion = BiEncoderLoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,} (Extremely Lightweight!)")
    
    # Mock Batch
    B = 4
    L_ag = 150
    L_ab = 60
    
    ag_feat = torch.randn(B, L_ag, config.antigen_dim)
    ab_feat = torch.randn(B, L_ab, config.antibody_dim)
    
    ag_mask = torch.ones(B, L_ag)
    ab_mask = torch.ones(B, L_ab)
    cdr_mask = torch.zeros(B, L_ab)
    cdr_mask[:, 20:35] = 1.0  # mock CDR region
    
    # Forward pass
    output = model(
        antigen_feat=ag_feat,
        antibody_feat=ab_feat,
        antigen_mask=ag_mask,
        antibody_mask=ab_mask,
        cdr_mask=cdr_mask
    )
    
    print("Output shapes:")
    print("  Antigen Embeddings:", output["antigen_embedding"].shape)
    print("  Antibody Embeddings:", output["antibody_embedding"].shape)
    print("  Paired Similarity:", output["paired_similarity"].shape)
    print("  Affinity Matrix:", output["affinity_matrix"].shape)
    
    # Loss calculation
    labels = torch.tensor([1.0, 1.0, 0.0, 1.0])
    loss = criterion(output["affinity_matrix"], labels)
    print(f"Computed Loss: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass: SUCCESS!")
    print("=== Verification Complete! ===")

if __name__ == "__main__":
    test_bi_encoder()

