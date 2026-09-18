from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

TOKENS_FILE = "tokens.f16"
INDEX_FILE = "index.json"


def sequence_hash(sequence: str) -> str:
    """Computes a 32-character SHA-256 hash for sequence keying."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:32]


class RaggedEmbeddingWriter:
    """Zero-copy memory-mapped embedding writer compatible with CALM/BEACON format."""
    def __init__(self, root: str | Path, dim: int, source: str = "custom") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.source = source
        self.entries: Dict[str, list[int]] = {}
        self.cursor = 0
        self.handle = open(self.root / TOKENS_FILE, "wb")

    def add(self, sequence: str, array: np.ndarray) -> None:
        key = sequence_hash(sequence)
        if key in self.entries:
            return
        block = np.ascontiguousarray(array, dtype=np.float16)
        if block.ndim != 2 or block.shape[1] != self.dim:
            raise ValueError(f"Expected (L, {self.dim}) embedding, got {block.shape}")
        block.tofile(self.handle)
        self.entries[key] = [self.cursor, int(block.shape[0])]
        self.cursor += int(block.shape[0])

    def close(self) -> None:
        self.handle.close()
        meta = {
            "dim": self.dim,
            "source": self.source,
            "tokens": self.cursor,
            "entries": self.entries,
        }
        (self.root / INDEX_FILE).write_text(json.dumps(meta), encoding="utf-8")

    def __enter__(self) -> "RaggedEmbeddingWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class MemoryMappedEmbeddingStore:
    """
    High-performance zero-copy memory-mapped embedding store.
    Adapted from CALM's ragged binary embedding format for ultra-fast I/O.
    """
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        index_path = self.root / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")
            
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        self.dim = int(meta["dim"])
        self.source = str(meta.get("source", "unknown"))
        self.entries = {
            key: (int(val[0]), int(val[1]))
            for key, val in meta["entries"].items()
        }
        self._mmap: Optional[np.ndarray] = None

    @property
    def data(self) -> np.ndarray:
        if self._mmap is None:
            token_file = self.root / TOKENS_FILE
            if not token_file.exists():
                raise FileNotFoundError(f"Missing token binary: {token_file}")
            flat = np.memmap(token_file, dtype=np.float16, mode="r")
            self._mmap = flat.reshape(-1, self.dim)
        return self._mmap

    def has(self, sequence: str) -> bool:
        return sequence_hash(sequence) in self.entries

    def get(self, sequence: str) -> np.ndarray:
        key = sequence_hash(sequence)
        if key not in self.entries:
            raise KeyError(f"Sequence not found in store: {key}")
        start, length = self.entries[key]
        return np.asarray(self.data[start : start + length])
