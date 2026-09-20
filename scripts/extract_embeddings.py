#!/usr/bin/env python3
"""
BEACON: Extract pre-computed residue embeddings for antibodies and antigens.
Uses AntiBERTy for antibodies (dim=512) and ESM-2 (650M) for antigens (dim=1280).
Saves to zero-copy memory-mapped store (tokens.f16 + index.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beacon.data.embeddings import RaggedEmbeddingWriter


def strip_special(array: np.ndarray, target_length: int) -> np.ndarray:
    if array.shape[0] == target_length:
        return array
    if array.shape[0] == target_length + 2:
        return array[1:-1]
    if array.shape[0] == target_length + 1:
        return array[1:]
    return array[:target_length]


class AntiBERTyExtractor:
    def __init__(self, device: torch.device):
        from antiberty import AntiBERTyRunner
        print(f"[+] Loading AntiBERTy model on {device}...")
        self.runner = AntiBERTyRunner()
        self.runner.device = device
        self.runner.model = self.runner.model.to(device).eval()
        self.dim = 512
        self.device = device

    def encode_batch(self, sequences: Sequence[str]) -> List[np.ndarray]:
        with torch.no_grad():
            embeddings = self.runner.embed(list(sequences))
        outputs = []
        for seq, tensor in zip(sequences, embeddings):
            arr = tensor.detach().float().cpu().numpy()
            outputs.append(strip_special(arr, len(seq)))
        return outputs


class ESM2Extractor:
    def __init__(self, device: torch.device, model_name: str = "esm2_t33_650M_UR50D"):
        import esm
        print(f"[+] Loading ESM-2 ({model_name}) on {device}...")
        model, alphabet = getattr(esm.pretrained, model_name)()
        self.model = model.eval().to(device)
        self.batch_converter = alphabet.get_batch_converter()
        self.repr_layer = int(model.num_layers)
        self.dim = int(model.embed_dim)
        self.device = device

    def encode_batch(self, sequences: Sequence[str]) -> List[np.ndarray]:
        batch = [(str(i), seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        hidden = out["representations"][self.repr_layer].float().cpu().numpy()
        return [hidden[i, 1:len(seq) + 1] for i, seq in enumerate(sequences)]


def extract_all(
    dataset_path: str,
    output_dir: str,
    device_str: str = "cuda",
    batch_size: int = 8
):
    df = pd.read_csv(dataset_path)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")

    ab_dir = out_p / "antibody"
    ag_dir = out_p / "antigen"

    # 1. Extract Antibodies (VH and VL)
    ab_seqs = sorted(list(set(
        [s for s in df["Antibody_VH_AA"].dropna().unique() if len(s) > 0] +
        [s for s in df["Antibody_VL_AA"].dropna().unique() if len(s) > 0]
    )))
    print(f"[+] Found {len(ab_seqs)} unique antibody chains to embed.")

    if not (ab_dir / "index.json").exists():
        ab_extractor = AntiBERTyExtractor(device)
        with RaggedEmbeddingWriter(ab_dir, dim=ab_extractor.dim, source="antiberty") as writer:
            for i in tqdm(range(0, len(ab_seqs), batch_size), desc="Embedding Antibodies"):
                batch = ab_seqs[i:i + batch_size]
                embeds = ab_extractor.encode_batch(batch)
                for seq, arr in zip(batch, embeds):
                    writer.add(seq, arr)
        print(f"[SUCCESS] Saved antibody embeddings to: {ab_dir}")
    else:
        print(f"[*] Antibody embeddings already exist in {ab_dir}, skipping.")

    # 2. Extract Antigens
    ag_seqs = sorted(list(set([s for s in df["Antigen_AA"].dropna().unique() if len(s) > 0])))
    print(f"[+] Found {len(ag_seqs)} unique antigens to embed.")

    if not (ag_dir / "index.json").exists():
        ag_extractor = ESM2Extractor(device)
        with RaggedEmbeddingWriter(ag_dir, dim=ag_extractor.dim, source="esm2_t33_650M_UR50D") as writer:
            for i in tqdm(range(0, len(ag_seqs), batch_size), desc="Embedding Antigens"):
                batch = ag_seqs[i:i + batch_size]
                embeds = ag_extractor.encode_batch(batch)
                for seq, arr in zip(batch, embeds):
                    writer.add(seq, arr)
        print(f"[SUCCESS] Saved antigen embeddings to: {ag_dir}")
    else:
        print(f"[*] Antigen embeddings already exist in {ag_dir}, skipping.")

    print("\n[COMPLETE] All residue embeddings extracted and ready for training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract residue embeddings for BEACON")
    parser.add_argument("--dataset", type=str, default="data/calm_dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--output-dir", type=str, default="data/embeddings", help="Output directory for embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding extraction")
    args = parser.parse_args()

    extract_all(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        device_str=args.device,
        batch_size=args.batch_size
    )

