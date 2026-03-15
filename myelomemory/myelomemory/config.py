"""Configuration dataclasses for MyeloMemory pipeline.

All hyperparameters, paths, and hardware settings are defined here.
Loaded from YAML via load_config().
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
import torch


@dataclass
class DataConfig:
    """Paths and parameters for data loading and preprocessing."""

    # Raw data directories (populated by scripts/download_data.sh)
    ccle_proteomics_path: Path = Path("data/raw/ccle_proteomics.csv")
    ccle_epigenomics_dir: Path = Path("data/raw/ccle_epigenomics/")
    string_ppi_path: Path = Path("data/raw/string_ppi.txt")
    gdsc_path: Path = Path("data/raw/gdsc_drug_sensitivity.csv")
    ctrpv2_path: Path = Path("data/raw/ctrpv2_drug_sensitivity.csv")
    kronke_path: Path = Path("data/raw/kronke_2024/")

    # Preprocessing
    min_protein_coverage: float = 0.7  # Drop proteins missing in >30% of samples
    imputation_method: str = "knn"  # knn | median | zero
    normalization: str = "quantile"  # quantile | zscore | log2
    ppi_confidence_threshold: float = 0.7  # STRING combined score cutoff

    # Splits
    test_fraction: float = 0.15
    val_fraction: float = 0.15
    random_seed: int = 42

    # Hematological lineage filter
    hematological_lineages: list[str] = field(default_factory=lambda: [
        "Myeloid", "Lymphoid",
        "haematopoietic_and_lymphoid_tissue",  # legacy CCLE name
    ])

    # Drug targets for GNN training
    target_drugs: list[str] = field(default_factory=lambda: [
        "Bortezomib", "Lenalidomide", "Dexamethasone",
        "Carfilzomib", "Pomalidomide", "Daratumumab",
    ])


@dataclass
class VAEConfig:
    """Hyperparameters for the Proteome-to-Epigenome conditional VAE."""

    # Architecture
    input_dim: int = 8000  # Number of proteins in feature vector
    epigenome_dim: int = 50000  # Number of ATAC-seq peaks to reconstruct
    latent_dim: int = 64  # Memory state embedding dimension
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "gelu"

    # Training
    pretrain_epochs: int = 200
    finetune_epochs: int = 100
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 512  # H100-optimized
    gradient_clip_norm: float = 1.0

    # KL annealing (cyclical)
    kl_anneal_cycles: int = 4
    kl_anneal_ratio: float = 0.5
    kl_weight_max: float = 1.0

    # Reconstruction loss weights
    atac_weight: float = 1.0
    h3k4me3_weight: float = 0.5
    h3k27me3_weight: float = 0.5

    # Gradient checkpointing (saves VRAM on H100 for large batches)
    gradient_checkpointing: bool = True

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4


@dataclass
class StabilityConfig:
    """Hyperparameters for the ODE-based Memory Stability Scorer."""

    # Chromatin reader/writer proteins to extract from proteome
    reader_writer_proteins: list[str] = field(default_factory=lambda: [
        "EZH2", "SUZ12", "EED",  # PRC2 writers (H3K27me3)
        "DNMT1", "DNMT3A", "DNMT3B",  # DNA methylation writers
        "TET1", "TET2", "TET3",  # DNA methylation erasers
        "KDM6A", "KDM6B",  # H3K27me3 erasers
        "SETD2",  # H3K36me3 writer
        "KMT2A", "KMT2B", "KMT2C", "KMT2D",  # H3K4me3 writers (MLL family)
        "KDM5A", "KDM5B",  # H3K4me3 erasers
        "HDAC1", "HDAC2", "HDAC3",  # Histone deacetylases
        "EP300", "CREBBP",  # Histone acetyltransferases
        "UHRF1",  # DNMT1 recruiter
        "SMARCB1", "SMARCA4",  # SWI/SNF remodelers
    ])

    # ODE parameters
    ode_solver: str = "dopri5"  # dopri5 | euler | rk4
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-7
    integration_time: float = 100.0  # Arbitrary time units for basin estimation
    n_perturbation_samples: int = 50  # Monte Carlo samples for basin depth

    # Calibration
    calibration_lr: float = 1e-3
    calibration_epochs: int = 500
    calibration_batch_size: int = 64

    # Output
    score_range: tuple[float, float] = (0.0, 1.0)  # Normalized stability score


@dataclass
class GNNConfig:
    """Hyperparameters for the Resistance Pathway GNN."""

    # Architecture
    node_feature_dim: int = 64 + 1 + 8000  # latent_dim + stability_score + proteomics
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8  # For GAT layers
    dropout: float = 0.2
    conv_type: str = "gat"  # gat | gcn | graphsage
    pool_type: str = "global_attention"  # global_attention | mean | max
    num_drugs: int = 6  # Number of drug targets
    predict_reversibility: bool = True  # Add binary reversibility head

    # Training
    epochs: int = 300
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 256  # H100-optimized
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"  # cosine | plateau | step

    # Loss
    resistance_loss_weight: float = 1.0
    reversibility_loss_weight: float = 0.5

    # Early stopping
    patience: int = 30
    min_delta: float = 1e-4


@dataclass
class HardwareConfig:
    """H100 GPU and distributed training settings."""

    device: str = "cuda"
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    compile: bool = True  # torch.compile
    compile_mode: str = "reduce-overhead"  # default | reduce-overhead | max-autotune
    pin_memory: bool = True
    num_workers: int = 8  # DataLoader workers
    prefetch_factor: int = 4

    # Distributed
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

    # Reproducibility
    deterministic: bool = False
    seed: int = 42


@dataclass
class APIConfig:
    """FastAPI server configuration."""

    port: int = 8000
    workers: int = 1
    max_batch_size: int = 64
    timeout: int = 30


@dataclass
class MyeloMemoryConfig:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    api: APIConfig = field(default_factory=APIConfig)

    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    wandb_project: str = "myelomemory"
    resume_checkpoint: Optional[Path] = None

    @property
    def device(self) -> torch.device:
        if self.hardware.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda", self.hardware.local_rank)
        return torch.device("cpu")

    @property
    def amp_dtype(self) -> torch.dtype:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.hardware.dtype, torch.bfloat16)


def load_config(path: Path) -> MyeloMemoryConfig:
    """Load configuration from a YAML file, with defaults for missing fields.

    Args:
        path: Path to YAML config file.

    Returns:
        Fully populated MyeloMemoryConfig.
    """
    if not path.exists():
        return MyeloMemoryConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = MyeloMemoryConfig()

    # Map nested YAML keys to dataclass fields
    section_map = {
        "data": (config.data, DataConfig),
        "vae": (config.vae, VAEConfig),
        "stability": (config.stability, StabilityConfig),
        "gnn": (config.gnn, GNNConfig),
        "hardware": (config.hardware, HardwareConfig),
        "api": (config.api, APIConfig),
    }

    for section_name, (section_obj, section_cls) in section_map.items():
        if section_name in raw:
            for key, value in raw[section_name].items():
                if hasattr(section_obj, key):
                    # Convert string paths to Path objects
                    field_type = section_cls.__dataclass_fields__[key].type
                    if field_type == Path or field_type == "Path":
                        value = Path(value)
                    setattr(section_obj, key, value)

    # Top-level fields
    for key in ("checkpoint_dir", "log_dir", "wandb_project"):
        if key in raw:
            value = raw[key]
            if key.endswith("_dir"):
                value = Path(value)
            setattr(config, key, value)

    return config
