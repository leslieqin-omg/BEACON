
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch

from beacon.config import BiEncoderConfig, TrainingConfig
from beacon.data.embeddings import MemoryMappedEmbeddingStore, RaggedEmbeddingWriter
from beacon.data.splits import nested_folds, assert_cluster_disjoint
from beacon.data.dataset import BiomoleculePairDataset
from scripts.train_kfold import train_single_fold, mean_and_ci95
from pathlib import Path

def test_kfold():
    print("[+] Creating temporary test environment...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create mock CSV
        N = 60
        clusters = [f"c_{i % 10}" for i in range(N)]
        df = pd.DataFrame({
            "Unique_AG_VH_VL_Hash": [f"pair_{i}" for i in range(N)],
            "Antibody_VH_AA": [f"EVQLVESGGGLVQPGGSLRLSCAASGFTF_{i % 10}" for i in range(N)],
            "Antibody_VL_AA": [f"DIQMTQSPSSLSASVGDRVTITCRASQSI_{i % 10}" for i in range(N)],
            "Antigen_AA": [f"MKTIIALSYIFCLVFA_{i % 10}" for i in range(N)],
            "Antibody_VH_Paratope": ["000000000000000000000000000000" for _ in range(N)],
            "Antibody_VL_Paratope": ["000000000000000000000000000000" for _ in range(N)],
            "Antigen_Epitope": ["0000000000000000" for _ in range(N)],
            "Cluster_id": clusters
        })
        csv_path = temp_dir / "test.csv"
        df.to_csv(csv_path, index=False)

        # Create mock embedding stores
        ab_dir = temp_dir / "embeddings" / "antibody"
        ag_dir = temp_dir / "embeddings" / "antigen"
        
        with RaggedEmbeddingWriter(ab_dir, dim=128) as w_ab:
            for vh in df["Antibody_VH_AA"].unique():
                w_ab.add(vh, np.random.randn(len(vh), 128).astype(np.float16))
            for vl in df["Antibody_VL_AA"].unique():
                w_ab.add(vl, np.random.randn(len(vl), 128).astype(np.float16))
                
        with RaggedEmbeddingWriter(ag_dir, dim=128) as w_ag:
            for ag in df["Antigen_AA"].unique():
                w_ag.add(ag, np.random.randn(len(ag), 128).astype(np.float16))

        ab_store = MemoryMappedEmbeddingStore(ab_dir)
        ag_store = MemoryMappedEmbeddingStore(ag_dir)

        dataset = BiomoleculePairDataset(df, ab_store, ag_store)
        splits = nested_folds(clusters, folds=5, seed=42)
        print(f"[+] Successfully generated {len(splits)} folds.")
        for s in splits:
            assert_cluster_disjoint(s, clusters)

        model_cfg = BiEncoderConfig(antigen_dim=128, antibody_dim=128, hidden_dim=64, projection_dim=32, num_conv_layers=1)
        train_cfg = TrainingConfig(batch_size=16, epochs=2, learning_rate=1e-3, device="cpu")

        # Test fold 0
        res = train_single_fold(
            fold_idx=0,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            full_dataset=dataset,
            split_indices=splits[0],
            device=torch.device("cpu"),
            fold_save_dir=temp_dir,
            ks=(1, 5, 10)
        )
        print("[+] Fold 0 finished successfully:", res["test"]["mean"])
        assert "mean" in res["test"]
        print("[SUCCESS] End-to-end k-fold test passed completely!")
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_kfold()
