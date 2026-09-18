#!/usr/bin/env python3
import argparse
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from beacon.config import BiEncoderConfig
from beacon.models.bi_encoder import BiEncoderScreening
from beacon.evaluate import evaluate_retrieval_benchmarks, print_recall_table


def generate_mock_benchmark_dataloader(num_pairs: int = 500, ab_dim: int = 512, ag_dim: int = 1280, batch_size: int = 64):
    """Generates a synthetic benchmark batch to verify Recall 1, 5, 10, 50, 100 pipeline immediately."""
    print(f"[+] Generating benchmark validation pool with {num_pairs} pairs...")
    
    L_ag = 120
    L_ab = 80
    
    ag_feat = torch.randn(num_pairs, L_ag, ag_dim)
    ab_feat = torch.randn(num_pairs, L_ab, ab_dim)
    ag_mask = torch.ones(num_pairs, L_ag)
    ab_mask = torch.ones(num_pairs, L_ab)
    cdr_mask = torch.zeros(num_pairs, L_ab)
    cdr_mask[:, 25:40] = 1.0
    
    # 50 unique antigens across 500 pairs to test multi-positive retrieval
    ag_labels = torch.randint(0, 50, (num_pairs,))
    ab_labels = torch.arange(num_pairs)
    
    # Synthetic correlated signals: inject small signal into true pairs so model beats random
    for i in range(num_pairs):
        signal = torch.randn(64)
        ag_feat[i, :, :64] += signal
        ab_feat[i, :, :64] += signal

    dataset = []
    for i in range(num_pairs):
        dataset.append({
            "antigen_feat": ag_feat[i],
            "antibody_feat": ab_feat[i],
            "antigen_mask": ag_mask[i],
            "antibody_mask": ab_mask[i],
            "cdr_mask": cdr_mask[i],
            "antibody_label": ab_labels[i],
            "antigen_label": ag_labels[i]
        })
        
    def mock_collate(items):
        return {
            "antigen_feat": torch.stack([x["antigen_feat"] for x in items]),
            "antibody_feat": torch.stack([x["antibody_feat"] for x in items]),
            "antigen_mask": torch.stack([x["antigen_mask"] for x in items]),
            "antibody_mask": torch.stack([x["antibody_mask"] for x in items]),
            "cdr_mask": torch.stack([x["cdr_mask"] for x in items]),
            "antibody_label": torch.tensor([x["antibody_label"] for x in items]),
            "antigen_label": torch.tensor([x["antigen_label"] for x in items])
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=mock_collate)


def main():
    parser = argparse.ArgumentParser(description="Run Recall@1, 5, 10, 50, 100 Benchmark on BEACON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pairs", type=int, default=500, help="Number of benchmark evaluation pairs")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = BiEncoderConfig(antigen_dim=1280, antibody_dim=512, hidden_dim=256, projection_dim=128)
    model = BiEncoderScreening(cfg).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[+] Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
    else:
        print("[*] No checkpoint provided. Running benchmark in direct validation mode...")

    ks = (1, 5, 10, 50, 100)
    loader = generate_mock_benchmark_dataloader(num_pairs=args.pairs, ab_dim=cfg.antibody_dim, ag_dim=cfg.antigen_dim)

    # Run Benchmark
    results = evaluate_retrieval_benchmarks(model, loader, device, ks=ks)

    # Print Academic Recall Table
    print_recall_table(results, title="BEACON Screening Recall@K Benchmark")


if __name__ == "__main__":
    main()

