# BEACON: Bi-Encoder Antibody-antigen Contrastive Orientation Network

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BEACON** is a high-throughput, sequence-based bi-encoder architecture designed for ultra-fast Stage-1 antibody-antigen candidate screening and retrieval. It decouples high-dimensional structural and sequence embeddings into compact, highly discriminative dual-tower projections with $O(1)$ cosine retrieval complexity.

---

## Key Highlights

- **Local Motif Modeling**: 1D Depthwise-Separable Convolutions ($k=5$) capture local contiguous sequence motifs, overcoming the single-residue limitation of point MLP baselines.
- **Gated Attentive Pooling**: Replaces uniform global mean pooling with learned soft attention, naturally upweighting functional CDR loops and epitope hot-spots while suppressing framework noise.
- **Memory-Mapped Zero-Copy I/O**: High-performance binary storage (`tokens.f16` + `index.json`) with OS-level `mmap`, streaming gigabytes of embeddings without RAM bottleneck.
- **Cluster-Disjoint Academic Benchmarks**: Nested 5-fold cross-validation under strict antigen sequence identity clustering thresholds (40%, 60%, 80%), eliminating data leakage.

---

## Architecture Overview

```
Antibody (VH + VL) ──> [ESM-2 / AntiBERTa2 Tokens] ──> [1D DW-Conv (k=5)] ──> [Gated Attention] ──> Proj ──> u_ab
                                                                                                                │ Cosine Sim
Antigen (Sequence)  ──> [ESM-2 Embeddings]         ──> [1D DW-Conv (k=5)] ──> [Gated Attention] ──> Proj ──> v_ag
```

---

## Installation

BEACON is built for ultra-fast reproduction on modern NVIDIA GPUs (e.g. RTX 4090 / A100) using `uv`:

```bash
# 1. Clone repository
git clone https://github.com/leslieqin-omg/BEACON.git
cd BEACON

# 2. Run automated environment setup
bash install_env.sh

# 3. Activate virtual environment
source .venv/bin/activate
```

---

## Training & 5-Fold Cross-Validation

To run 5-fold cross validation on disjoint antigen clusters:

```bash
# Train on 40% sequence identity cluster splits
python scripts/train_kfold.py --config configs/ag40.yaml

# Train on 60% or 80% identity cluster splits
python scripts/train_kfold.py --config configs/ag60.yaml
python scripts/train_kfold.py --config configs/ag80.yaml
```

---

## Benchmark Figures & Academic Reporting

Generate publication-grade 300 DPI figures and summary tables:

```bash
python scripts/plot_benchmarks.py \
    --runs-dir artifacts/runs \
    --output-dir artifacts/report
```

### Retrieval Performance Across Identity Splits

| Split | Setting | Folds | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AG40** | Ab-Ag 40% Disjoint | 5-Fold | 0.96 ± 0.31% | 4.10 ± 0.92% | 7.79 ± 1.38% | 30.82 ± 4.26% | 52.11 ± 4.91% |
| **AG60** | Ab-Ag 60% Disjoint | 5-Fold | 2.17 ± 0.60% | 6.17 ± 1.87% | 10.74 ± 2.39% | 34.00 ± 4.98% | 54.06 ± 6.29% |
| **AG80** | Ab-Ag 80% Disjoint | 5-Fold | 5.18 ± 1.53% | 11.11 ± 3.14% | 15.23 ± 3.71% | 35.27 ± 3.54% | 56.02 ± 3.56% |

---

## Citation & Contact

```bibtex
@article{beacon2026,
  title={BEACON: Bi-Encoder Antibody-antigen Contrastive Orientation Network for High-Throughput Screening},
  author={Qin, Haifeng and Collaborators},
  year={2026}
}
```

