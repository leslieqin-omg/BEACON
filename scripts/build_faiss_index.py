#!/usr/bin/env python3
"""
BEACON Offline FAISS Vector Indexing & Query Utility.
Builds an offline vector database from antibody libraries using trained BEACON checkpoints,
enabling sub-millisecond candidate screening.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Setup project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beacon.config import BiEncoderConfig
from beacon.models.bi_encoder import BiEncoderScreening
from beacon.data.dataset import BiomoleculePairDataset, collate_padded_batch
from beacon.data.embeddings import H5EmbeddingStore


def build_faiss_index_from_embeddings(
    vectors: np.ndarray,
    index_type: str = "FlatIP",
    hnsw_m: int = 32
):
    """
    Builds a FAISS index from L2-normalized float32 vectors.
    index_type: 'FlatIP' (exact inner-product / cosine similarity)
                'HNSW' (Hierarchical Navigable Small World for millions of candidates)
    """
    try:
        import faiss
    except ImportError:
        print("[!] Warning: faiss is not installed. To use native FAISS indexing, run:")
        print("    uv pip install faiss-cpu  (or faiss-gpu)")
        return None

    dim = vectors.shape[1]
    # Ensure float32 and contiguous
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if index_type.upper() == "HNSW":
        print(f"[+] Creating FAISS HNSW Index (dim={dim}, M={hnsw_m}, metric=INNER_PRODUCT)...")
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 64
        index.hnsw.efSearch = 32
    else:
        print(f"[+] Creating FAISS FlatIP Index (dim={dim}, exact cosine similarity)...")
        index = faiss.IndexFlatIP(dim)

    index.add(vectors)
    print(f"[+] Successfully added {index.ntotal} vectors to FAISS index.")
    return index


def main():
    parser = argparse.ArgumentParser(description="Build and Query BEACON Offline FAISS Index")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained BEACON checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="data/calm_dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings", help="Path to precomputed PLM embeddings")
    parser.add_argument("--output-dir", type=str, default="artifacts/index", help="Output directory for index & metadata")
    parser.add_argument("--index-type", type=str, default="FlatIP", choices=["FlatIP", "HNSW"], help="FAISS index type")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--demo-queries", type=int, default=5, help="Number of benchmark query tests to run after building")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("  BEACON Offline Vector Index Construction")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Target Device: {device} | Index Type: {args.index_type}")
    print("==========================================================================")

    # 1. Load trained checkpoint
    print(f"[+] Loading model checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Extract config
    model_state = ckpt.get("model_state", ckpt)
    cfg_data = ckpt.get("config", {})
    if isinstance(cfg_data, dict) and "model" in cfg_data:
        cfg = BiEncoderConfig(**cfg_data["model"])
    elif isinstance(cfg_data, BiEncoderConfig):
        cfg = cfg_data
    else:
        cfg = BiEncoderConfig()

    model = BiEncoderScreening(cfg).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # 2. Load dataset
    print(f"[+] Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    ab_store = H5EmbeddingStore(Path(args.embedding_dir) / "antibody")
    ag_store = H5EmbeddingStore(Path(args.embedding_dir) / "antigen")
    dataset = BiomoleculePairDataset(df, ab_store, ag_store)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_padded_batch,
        num_workers=(4 if device.type == "cuda" else 0)
    )

    # 3. Extract 128-d L2-normalized embeddings for the entire library
    print(f"[+] Encoding {len(dataset)} antibody candidates using BEACON Antibody Tower...")
    all_ab_vectors = []
    all_ag_vectors = []
    meta_records = []

    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            ab_feat = batch["antibody_feat"].to(device, non_blocking=True)
            ab_mask = batch["antibody_mask"].to(device, non_blocking=True)
            ag_feat = batch["antigen_feat"].to(device, non_blocking=True)
            ag_mask = batch["antigen_mask"].to(device, non_blocking=True)

            # BEACON projection
            u_ab = model.encode_antibody(ab_feat, ab_mask)
            v_ag = model.encode_antigen(ag_feat, ag_mask)

            all_ab_vectors.append(u_ab.cpu().numpy())
            all_ag_vectors.append(v_ag.cpu().numpy())

    ab_matrix = np.concatenate(all_ab_vectors, axis=0).astype(np.float32)
    ag_matrix = np.concatenate(all_ag_vectors, axis=0).astype(np.float32)
    elapsed = time.time() - start_time
    print(f"[+] Finished encoding in {elapsed:.2f}s ({len(dataset) / elapsed:.1f} pairs/sec).")

    # 4. Save metadata
    for idx, row in df.iterrows():
        meta_records.append({
            "index_id": idx,
            "antibody_id": getattr(row, "Antibody_ID", f"Ab_{idx}"),
            "antigen_id": getattr(row, "Antigen_ID", f"Ag_{idx}"),
            "antibody_vh": getattr(row, "Antibody_VH", ""),
            "antigen_seq": getattr(row, "Antigen_Seq", ""),
            "cluster_id": getattr(row, "Antigen_Cluster", -1)
        })

    meta_path = out_dir / "antibody_metadata.json"
    meta_path.write_text(json.dumps(meta_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Saved metadata to: {meta_path}")

    # Save raw numpy embeddings as backup
    np_path = out_dir / "antibody_embeddings.npy"
    np.save(np_path, ab_matrix)
    print(f"[+] Saved raw L2-normalized numpy vectors to: {np_path}")

    # 5. Build FAISS index if installed
    index = build_faiss_index_from_embeddings(ab_matrix, index_type=args.index_type)
    if index is not None:
        try:
            import faiss
            index_path = out_dir / f"antibody_library_{args.index_type.lower()}.index"
            faiss.write_index(index, str(index_path))
            print(f"[+] Saved FAISS index file to: {index_path}")
        except Exception as e:
            print(f"[!] Could not serialize FAISS index: {e}")

    # 6. Benchmark Online Query Latency
    if args.demo_queries > 0:
        print("\n--------------------------------------------------------------------------")
        print("  Online Single-Target Query Latency Benchmark (Sub-millisecond test)")
        print("--------------------------------------------------------------------------")
        num_test = min(args.demo_queries, len(ag_matrix))
        test_queries = ag_matrix[:num_test]

        if index is not None:
            # Query using native FAISS
            t0 = time.perf_counter()
            distances, indices = index.search(test_queries, k=10)
            t_total = (time.perf_counter() - t0) * 1000.0  # ms
            avg_lat = t_total / num_test
            print(f"[FAISS Search] Queried Top-10 over {len(ab_matrix)} candidates:")
            print(f"               Average Latency per target: {avg_lat:.3f} ms")
            for q_idx in range(min(3, num_test)):
                matched_id = indices[q_idx][0]
                matched_score = distances[q_idx][0]
                print(f"  Target #{q_idx}: Top-1 Matched Candidate Index={matched_id}, Cosine Sim={matched_score:.4f}")
        else:
            # Fallback numpy exact dot product benchmark
            t0 = time.perf_counter()
            sims = np.dot(test_queries, ab_matrix.T)
            top10 = np.argsort(-sims, axis=1)[:, :10]
            t_total = (time.perf_counter() - t0) * 1000.0
            avg_lat = t_total / num_test
            print(f"[NumPy Search] Queried Top-10 over {len(ab_matrix)} candidates:")
            print(f"               Average Latency per target: {avg_lat:.3f} ms")

    print("\n[SUCCESS] Offline vector indexing workflow completed successfully.")


if __name__ == "__main__":
    main()

