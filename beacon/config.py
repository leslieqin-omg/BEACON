from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class BiEncoderConfig:
    # Input feature dimensions (e.g., from ESM-2 or light sequence models)
    antigen_dim: int = 1280       # ESM-2 650M is 1280, ESM-2 35M is 480, ESM-2 8M is 320
    antibody_dim: int = 512       # AntiBERTy is 512, ESM-2 8M is 320
    
    # Projection & Embedding space
    hidden_dim: int = 256
    projection_dim: int = 128     # Final embedding dimension for vector indexing / cosine similarity
    
    # Encoder architecture
    num_conv_layers: int = 2      # Local 1D depthwise-separable conv layers
    kernel_size: int = 5
    dropout: float = 0.1
    
    # Pooling strategy
    antibody_pooling: str = "cdr_attention"  # "cdr_attention", "cdr_mean", "mean"
    antigen_pooling: str = "attention"       # "attention", "mean"
    
    # Similarity and Scaling
    init_temperature: float = 0.07           # Learnable InfoNCE temperature (1/tau)
    learnable_temperature: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    device: str = "cuda"


def load_config(path: str) -> BiEncoderConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BiEncoderConfig(**data.get("model", {}))
@dataclass
class EvaluationConfig:
    pool_size: int = 256
    test_pairs_per_fold: int = 512
    ks: tuple = (1, 5, 10, 50, 100)


@dataclass
class KFoldExperimentConfig:
    name: str = "ag40"
    label: str = "Ab-Ag 40% Cluster Disjoint"
    folds: int = 5
    min_seq_id: float = 0.4
    cluster_on: str = "antigen"
    output_dir: str = "artifacts/runs"
    model: BiEncoderConfig = field(default_factory=BiEncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_kfold_config(path: str) -> KFoldExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    model_cfg = BiEncoderConfig(**data.get("model", {}))
    train_cfg = TrainingConfig(**data.get("training", {}))
    eval_cfg = EvaluationConfig(**data.get("evaluation", {}))
    return KFoldExperimentConfig(
        name=data.get("name", "ag40"),
        label=data.get("label", "Ab-Ag 40% Cluster Disjoint"),
        folds=int(data.get("folds", 5)),
        min_seq_id=float(data.get("min_seq_id", 0.4)),
        cluster_on=data.get("cluster_on", "antigen"),
        output_dir=data.get("output_dir", "artifacts/runs"),
        model=model_cfg,
        training=train_cfg,
        evaluation=eval_cfg
    )
