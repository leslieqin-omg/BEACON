from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from beacon.data.embeddings import MemoryMappedEmbeddingStore


@dataclass
class PairSample:
    pair_id: str
    vh: str
    vl: str
    antigen: str
    vh_cdr_mask: np.ndarray
    vl_cdr_mask: np.ndarray
    antigen_mask: np.ndarray
    antibody_label: int
    antigen_label: int
    cluster_id: str


def parse_mask_string(mask_str: Any, length: int) -> np.ndarray:
    """Parses binary mask string (e.g. '000111000') into boolean numpy array."""
    if not isinstance(mask_str, str) or len(mask_str) == 0:
        return np.ones(length, dtype=bool)
    arr = np.array([ch == "1" for ch in mask_str[:length]], dtype=bool)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), constant_values=False)
    return arr


class BiomoleculePairDataset(Dataset):
    """
    Dataset for paired antibody-antigen screening.
    Supports on-the-fly CDR slicing or full sequence with attention weighting.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        antibody_store: MemoryMappedEmbeddingStore,
        antigen_store: MemoryMappedEmbeddingStore,
        focus_on_interface: bool = True
    ) -> None:
        self.antibody_store = antibody_store
        self.antigen_store = antigen_store
        self.focus_on_interface = focus_on_interface
        
        # Build index maps for multi-positive contrastive labels
        ab_to_label: Dict[str, int] = {}
        ag_to_label: Dict[str, int] = {}
        
        self.samples: List[PairSample] = []
        for row in dataframe.itertuples():
            vh_seq = getattr(row, "Antibody_VH_AA", "")
            vl_seq = getattr(row, "Antibody_VL_AA", "")
            ag_seq = getattr(row, "Antigen_AA", "")
            
            vh_mask = parse_mask_string(getattr(row, "Antibody_VH_Paratope", ""), len(vh_seq))
            vl_mask = parse_mask_string(getattr(row, "Antibody_VL_Paratope", ""), len(vl_seq))
            ag_mask = parse_mask_string(getattr(row, "Antigen_Epitope", ""), len(ag_seq))
            
            ab_key = f"{vh_seq}/{vl_seq}"
            ag_key = ag_seq
            
            if ab_key not in ab_to_label:
                ab_to_label[ab_key] = len(ab_to_label)
            if ag_key not in ag_to_label:
                ag_to_label[ag_key] = len(ag_to_label)
                
            pair_id = str(getattr(row, "Unique_AG_VH_VL_Hash", f"pair_{len(self.samples)}"))
            cluster_id = str(getattr(row, "Cluster_id", "c0"))
            
            self.samples.append(
                PairSample(
                    pair_id=pair_id,
                    vh=vh_seq,
                    vl=vl_seq,
                    antigen=ag_seq,
                    vh_cdr_mask=vh_mask,
                    vl_cdr_mask=vl_mask,
                    antigen_mask=ag_mask,
                    antibody_label=ab_to_label[ab_key],
                    antigen_label=ag_to_label[ag_key],
                    cluster_id=cluster_id
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        
        # 1. Fetch Antibody tokens
        ab_tokens_list = []
        ab_cdr_list = []
        for seq, mask in ((s.vh, s.vh_cdr_mask), (s.vl, s.vl_cdr_mask)):
            if seq and self.antibody_store.has(seq):
                toks = self.antibody_store.get(seq)
                limit = min(len(toks), len(mask))
                ab_tokens_list.append(toks[:limit])
                ab_cdr_list.append(mask[:limit])
                
        if ab_tokens_list:
            ab_tokens = np.concatenate(ab_tokens_list, axis=0)
            ab_cdr = np.concatenate(ab_cdr_list, axis=0)
        else:
            ab_tokens = np.zeros((1, self.antibody_store.dim), dtype=np.float16)
            ab_cdr = np.zeros((1,), dtype=bool)

        # 2. Fetch Antigen tokens
        if s.antigen and self.antigen_store.has(s.antigen):
            ag_tokens = self.antigen_store.get(s.antigen)
            limit = min(len(ag_tokens), len(s.antigen_mask))
            ag_tokens = ag_tokens[:limit]
            ag_epi = s.antigen_mask[:limit]
        else:
            ag_tokens = np.zeros((1, self.antigen_store.dim), dtype=np.float16)
            ag_epi = np.zeros((1,), dtype=bool)

        return {
            "ab_tokens": ab_tokens,
            "ab_cdr": ab_cdr,
            "ag_tokens": ag_tokens,
            "ag_epi": ag_epi,
            "antibody_label": s.antibody_label,
            "antigen_label": s.antigen_label,
            "pair_id": s.pair_id
        }


def collate_padded_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pads variable-length sequences to form regular batch tensors for high GPU utilization."""
    B = len(batch)
    
    max_ab_len = max(item["ab_tokens"].shape[0] for item in batch)
    max_ag_len = max(item["ag_tokens"].shape[0] for item in batch)
    
    ab_dim = batch[0]["ab_tokens"].shape[1]
    ag_dim = batch[0]["ag_tokens"].shape[1]
    
    ab_feat = torch.zeros(B, max_ab_len, ab_dim, dtype=torch.float32)
    ag_feat = torch.zeros(B, max_ag_len, ag_dim, dtype=torch.float32)
    
    ab_mask = torch.zeros(B, max_ab_len, dtype=torch.float32)
    ag_mask = torch.zeros(B, max_ag_len, dtype=torch.float32)
    
    cdr_mask = torch.zeros(B, max_ab_len, dtype=torch.float32)
    epi_mask = torch.zeros(B, max_ag_len, dtype=torch.float32)
    
    ab_labels = []
    ag_labels = []
    
    for i, item in enumerate(batch):
        ab_len = item["ab_tokens"].shape[0]
        ag_len = item["ag_tokens"].shape[0]
        
        ab_feat[i, :ab_len] = torch.from_numpy(item["ab_tokens"].astype(np.float32))
        ag_feat[i, :ag_len] = torch.from_numpy(item["ag_tokens"].astype(np.float32))
        
        ab_mask[i, :ab_len] = 1.0
        ag_mask[i, :ag_len] = 1.0
        
        cdr_mask[i, :ab_len] = torch.from_numpy(item["ab_cdr"].astype(np.float32))
        epi_mask[i, :ag_len] = torch.from_numpy(item["ag_epi"].astype(np.float32))
        
        ab_labels.append(item["antibody_label"])
        ag_labels.append(item["antigen_label"])
        
    return {
        "antigen_feat": ag_feat,
        "antibody_feat": ab_feat,
        "antigen_mask": ag_mask,
        "antibody_mask": ab_mask,
        "cdr_mask": cdr_mask,
        "epitope_mask": epi_mask,
        "antibody_label": torch.tensor(ab_labels, dtype=torch.long),
        "antigen_label": torch.tensor(ag_labels, dtype=torch.long),
    }


def create_dataloader(
    dataset: BiomoleculePairDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_padded_batch,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0)
    )

