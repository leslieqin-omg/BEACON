#!/usr/bin/env python3
import argparse
import pandas as pd
import yaml

from beacon.config import BiEncoderConfig, TrainingConfig
from beacon.data.embeddings import MemoryMappedEmbeddingStore
from beacon.data.dataset import BiomoleculePairDataset, create_dataloader
from beacon.train import train_screening_model


def main():
    parser = argparse.ArgumentParser(description="Train BEACON Bi-Encoder Screening Model on RTX 4090")
    parser.add_argument("--config", type=str, default="configs/rtx4090.yaml", help="Path to config YAML")
    parser.add_argument("--data-csv", type=str, default=None, help="Override path to dataset CSV")
    parser.add_argument("--emb-dir", type=str, default=None, help="Override path to embeddings root")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = BiEncoderConfig(**cfg_dict.get("model", {}))
    train_cfg = TrainingConfig(**cfg_dict.get("training", {}))
    data_cfg = cfg_dict.get("data", {})

    csv_path = args.data_csv or data_cfg.get("dataset_csv", "data/calm_dataset.csv")
    emb_dir = args.emb_dir or data_cfg.get("embedding_dir", "artifacts/embeddings")

    print(f"[+] Loading dataset from: {csv_path}")
    print(f"[+] Loading embeddings from: {emb_dir}")

    # Load zero-copy memory mapped embedding stores
    ab_store = MemoryMappedEmbeddingStore(f"{emb_dir}/antibody")
    ag_store = MemoryMappedEmbeddingStore(f"{emb_dir}/antigen")

    df = pd.read_csv(csv_path)
    # 80/20 train/val split by cluster or random
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = BiomoleculePairDataset(train_df, ab_store, ag_store)
    val_dataset = BiomoleculePairDataset(val_df, ab_store, ag_store)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True)
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True)
    )

    print(f"[+] Starting training: {len(train_dataset)} train pairs, {len(val_dataset)} val pairs.")
    train_screening_model(model_cfg, train_cfg, train_loader, val_loader)


if __name__ == "__main__":
    main()

