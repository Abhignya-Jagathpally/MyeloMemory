#!/usr/bin/env python3
"""MyeloMemory — AI pipeline for epigenetic drug resistance memory prediction.

Entry point. Orchestrates the full pipeline through clean function calls.
Each stage checks for existing checkpoints and skips if already completed.

Usage:
    python main.py --config configs/h100.yaml
    python main.py --config configs/h100.yaml --stage vae_pretrain
    python main.py --config configs/h100.yaml --resume-from-latest
    torchrun --nproc_per_node=8 main.py --config configs/h100.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from myelomemory.config import load_config, MyeloMemoryConfig
from myelomemory.utils.checkpoint import CheckpointManager
from myelomemory.utils.logging import setup_logger, log_stage_start, log_stage_end


# ---------------------------------------------------------------------------
# Stage functions — each does ONE thing, returns a checkpoint path
# ---------------------------------------------------------------------------

def validate_data_files(config: MyeloMemoryConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Validate that all required raw data files exist before any processing.

    Checks for every file the pipeline needs and fails fast with
    actionable download instructions if anything is missing.
    This runs BEFORE data preprocessing and has no fallbacks.

    Raises:
        FileNotFoundError: With specific instructions for each missing file.
    """
    log_stage_start("data_validate")

    missing = []

    # Check proteomics
    if not config.data.ccle_proteomics_path.exists():
        missing.append(
            f"  - CCLE Proteomics: {config.data.ccle_proteomics_path}\n"
            f"    Download from: https://depmap.org/portal/download/all/"
        )

    # Check sample metadata (for lineage labels)
    metadata_path = config.data.ccle_proteomics_path.parent / "sample_info.csv"
    if not metadata_path.exists():
        missing.append(
            f"  - CCLE Sample Metadata: {metadata_path}\n"
            f"    Download 'Model.csv' from: https://depmap.org/portal/download/all/"
        )

    # Check epigenomics (at least one file required)
    epi_dir = config.data.ccle_epigenomics_dir
    epi_files = ["chromatin_profiling.csv", "atac_seq.csv", "h3k4me3.csv", "h3k27me3.csv"]
    epi_found = [f for f in epi_files if (epi_dir / f).exists()]
    if not epi_found:
        missing.append(
            f"  - Epigenomic data: at least one of {epi_files} in {epi_dir}\n"
            f"    Chromatin profiling or ATAC-seq from DepMap; histone ChIP-seq from ENCODE"
        )

    # Check STRING PPI
    if not config.data.string_ppi_path.exists():
        missing.append(
            f"  - STRING PPI: {config.data.string_ppi_path}\n"
            f"    Download from: https://string-db.org/cgi/download (organism: 9606)\n"
            f"    Convert Ensembl IDs to gene symbols before use."
        )

    # Check drug sensitivity (at least one source required)
    gdsc_exists = config.data.gdsc_path.exists()
    ctrpv2_exists = config.data.ctrpv2_path.exists()
    if not gdsc_exists and not ctrpv2_exists:
        missing.append(
            f"  - Drug sensitivity: at least one of:\n"
            f"      GDSC: {config.data.gdsc_path} (from https://www.cancerrxgene.org/)\n"
            f"      CTRPv2: {config.data.ctrpv2_path} (from Broad Institute)"
        )

    if missing:
        msg = (
            "Required data files are missing. The pipeline requires real "
            "datasets and has no synthetic fallbacks.\n\n"
            "Missing files:\n" + "\n".join(missing) + "\n\n"
            "Run 'bash scripts/download_data.sh' for detailed download instructions."
        )
        raise FileNotFoundError(msg)

    log_stage_end("data_validate")
    return Path("validated")


def prepare_data(config: MyeloMemoryConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Preprocess and harmonize all multi-omics datasets.

    Requires all raw data files to be present (run validate_data_files first
    or bash scripts/download_data.sh). No synthetic fallbacks.

    Outputs:
        checkpoints/data_ready.pt — preprocessed tensors, split indices,
        normalization parameters.
    """
    from myelomemory.data.loaders import load_ccle_proteomics, load_ccle_epigenomics, load_string_ppi
    from myelomemory.data.preprocessors import harmonize_omics, build_train_val_test_splits

    if ckpt_mgr.exists("data_ready"):
        return ckpt_mgr.path("data_ready")

    log_stage_start("data_prep")

    proteomics = load_ccle_proteomics(config.data)
    epigenomics = load_ccle_epigenomics(config.data)
    ppi_graph = load_string_ppi(config.data)

    dataset = harmonize_omics(proteomics, epigenomics, ppi_graph, config.data)
    splits = build_train_val_test_splits(dataset, config.data)

    ckpt_path = ckpt_mgr.save("data_ready", {
        "dataset": dataset,
        "splits": splits,
        "config": config.data,
    })

    log_stage_end("data_prep")
    return ckpt_path


def pretrain_vae(config: MyeloMemoryConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Pretrain the conditional VAE on pan-cancer CCLE data.

    Uses all ~1,000 CCLE cell lines to learn a general proteome→epigenome
    mapping before fine-tuning on the hematological subset.

    Outputs:
        checkpoints/vae_pretrained.pt — encoder/decoder weights, optimizer state.
    """
    from myelomemory.models.vae import ProteomeToEpigenomeVAE, train_vae

    if ckpt_mgr.exists("vae_pretrained"):
        return ckpt_mgr.path("vae_pretrained")

    log_stage_start("vae_pretrain")

    data_ckpt = ckpt_mgr.load("data_ready")
    dataset = data_ckpt["dataset"]

    # Override VAE dimensions from actual data shape
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]

    model = ProteomeToEpigenomeVAE(config.vae).to(config.device)
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)

    result = train_vae(
        model=model,
        dataset=dataset,
        splits=data_ckpt["splits"],
        config=config.vae,
        subset="pan_cancer",
        ckpt_mgr=ckpt_mgr,
        stage_name="vae_pretrained",
    )

    log_stage_end("vae_pretrain", metrics=result["metrics"])
    return result["checkpoint_path"]


def finetune_vae(config: MyeloMemoryConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Fine-tune the VAE on hematological cell lines only.

    Loads pretrained weights and continues training on the ~50 hematological
    lines with matched proteomic + epigenomic data.

    Outputs:
        checkpoints/vae_finetuned.pt
    """
    from myelomemory.models.vae import ProteomeToEpigenomeVAE, train_vae

    if ckpt_mgr.exists("vae_finetuned"):
        return ckpt_mgr.path("vae_finetuned")

    log_stage_start("vae_finetune")

    data_ckpt = ckpt_mgr.load("data_ready")
    pretrained = ckpt_mgr.load("vae_pretrained")
    dataset = data_ckpt["dataset"]

    # Override VAE dimensions from actual data shape
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]

    model = ProteomeToEpigenomeVAE(config.vae).to(config.device)
    model.load_state_dict(pretrained["model_state_dict"])
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)

    result = train_vae(
        model=model,
        dataset=dataset,
        splits=data_ckpt["splits"],
        config=config.vae,
        subset="hematological",
        ckpt_mgr=ckpt_mgr,
        stage_name="vae_finetuned",
    )

    log_stage_end("vae_finetune", metrics=result["metrics"])
    return result["checkpoint_path"]


def calibrate_stability(config: MyeloMemoryConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Calibrate the ODE-based memory stability scorer.

    Parameterizes the Sneppen-Ringrose bistability model with chromatin
    reader/writer protein levels from the proteomic profiles, then calibrates
    basin-of-attraction depth against CCLE washout time-course data.

    Outputs:
        checkpoints/stability_calibrated.pt
    """
    from myelomemory.models.stability import MemoryStabilityScorer, calibrate_scorer

    if ckpt_mgr.exists("stability_calibrated"):
        return ckpt_mgr.path("stability_calibrated")

    log_stage_start("stability_calibrate")

    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")

    scorer = MemoryStabilityScorer(config.stability).to(config.device)

    result = calibrate_scorer(
        scorer=scorer,
        dataset=data_ckpt["dataset"],
        vae_checkpoint=vae_ckpt,
        config=config.stability,
        ckpt_mgr=ckpt_mgr,
    )

    log_stage_end("stability_calibrate", metrics=result["metrics"])
    return result["checkpoint_path"]


def train_gnn(config: MyeloMemoryConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Train the GNN drug resistance classifier on STRING PPI network.

    Node features: protein abundance + inferred memory state + stability score.
    Training target: per-drug IC50 from GDSC/CTRPv2.

    Outputs:
        checkpoints/gnn_trained.pt
    """
    from myelomemory.models.gnn import ResistanceGNN, train_resistance_gnn

    if ckpt_mgr.exists("gnn_trained"):
        return ckpt_mgr.path("gnn_trained")

    log_stage_start("gnn_train")

    data_ckpt = ckpt_mgr.load("data_ready")
    vae_ckpt = ckpt_mgr.load("vae_finetuned")
    stability_ckpt = ckpt_mgr.load("stability_calibrated")
    dataset = data_ckpt["dataset"]

    # Override dimensions from actual data
    config.vae.input_dim = dataset.proteomics.shape[1]
    config.vae.epigenome_dim = dataset.epigenomics.shape[1]
    # Node features: 1 (protein abundance) + latent_dim (memory state) + 1 (stability score)
    config.gnn.node_feature_dim = 1 + config.vae.latent_dim + 1
    config.gnn.num_drugs = dataset.drug_sensitivity.shape[1]

    model = ResistanceGNN(config.gnn).to(config.device)
    model = _maybe_compile(model, config)
    model = _maybe_distribute(model, config)

    result = train_resistance_gnn(
        model=model,
        dataset=dataset,
        splits=data_ckpt["splits"],
        vae_checkpoint=vae_ckpt,
        stability_checkpoint=stability_ckpt,
        config=config.gnn,
        ckpt_mgr=ckpt_mgr,
        vae_config=config.vae,
        stability_config=config.stability,
    )

    log_stage_end("gnn_train", metrics=result["metrics"])
    return result["checkpoint_path"]


def validate_pipeline(config: MyeloMemoryConfig, ckpt_mgr: CheckpointManager) -> Path:
    """Run end-to-end validation on held-out data.

    Loads all three trained modules, runs inference on the validation set,
    computes AUROC, AUPRC, and the novel resistance-reversibility metric.

    Outputs:
        checkpoints/pipeline_validated.pt
    """
    from myelomemory.inference.pipeline import MyeloMemoryPipeline
    from myelomemory.utils.metrics import compute_full_metrics

    if ckpt_mgr.exists("pipeline_validated"):
        return ckpt_mgr.path("pipeline_validated")

    log_stage_start("validate")

    pipeline = MyeloMemoryPipeline.from_checkpoints(ckpt_mgr, config)
    data_ckpt = ckpt_mgr.load("data_ready")

    predictions = pipeline.predict(data_ckpt["dataset"], split="test")
    metrics = compute_full_metrics(predictions, data_ckpt["dataset"], split="test")

    ckpt_path = ckpt_mgr.save("pipeline_validated", {
        "metrics": metrics,
        "config": config,
    })

    log_stage_end("validate", metrics=metrics)
    return ckpt_path


def serve_api(config: MyeloMemoryConfig, ckpt_mgr: CheckpointManager) -> None:
    """Launch the FastAPI inference server.

    Loads the validated pipeline and serves predictions via REST API.
    """
    from myelomemory.inference.api import create_app
    import uvicorn

    log_stage_start("serve")

    app = create_app(ckpt_mgr, config)
    uvicorn.run(app, host="0.0.0.0", port=config.api.port, workers=config.api.workers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_compile(model: torch.nn.Module, config: MyeloMemoryConfig) -> torch.nn.Module:
    """Apply torch.compile if enabled in config (requires PyTorch 2.0+)."""
    if config.hardware.compile:
        return torch.compile(model, mode=config.hardware.compile_mode)
    return model


def _maybe_distribute(model: torch.nn.Module, config: MyeloMemoryConfig) -> torch.nn.Module:
    """Wrap in DDP if running in distributed mode."""
    if config.hardware.distributed and dist.is_initialized():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.hardware.local_rank],
            output_device=config.hardware.local_rank,
        )
    return model


def _init_distributed(config: MyeloMemoryConfig) -> None:
    """Initialize distributed training if WORLD_SIZE > 1."""
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        config.hardware.distributed = True
        config.hardware.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        config.hardware.world_size = dist.get_world_size()
        torch.cuda.set_device(config.hardware.local_rank)


# ---------------------------------------------------------------------------
# Pipeline stages registry — ordered list of (name, function)
# ---------------------------------------------------------------------------

STAGES = [
    ("data_validate",        validate_data_files),
    ("data_prep",            prepare_data),
    ("vae_pretrain",         pretrain_vae),
    ("vae_finetune",         finetune_vae),
    ("stability_calibrate",  calibrate_stability),
    ("gnn_train",            train_gnn),
    ("validate",             validate_pipeline),
    ("serve",                serve_api),
]


def run_pipeline(config: MyeloMemoryConfig, stage: str | None = None) -> None:
    """Execute the full pipeline or a single stage.

    If stage is None, runs all stages sequentially, skipping any that
    have existing checkpoints. If stage is specified, runs only that stage.
    """
    logger = setup_logger(config)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, logger)

    _init_distributed(config)

    # H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if config.hardware.deterministic:
        torch.use_deterministic_algorithms(True)

    if stage:
        stage_fn = dict(STAGES).get(stage)
        if stage_fn is None:
            valid = [name for name, _ in STAGES]
            logger.error(f"Unknown stage '{stage}'. Valid stages: {valid}")
            sys.exit(1)
        stage_fn(config, ckpt_mgr)
    else:
        for name, fn in STAGES:
            logger.info(f"=== Pipeline stage: {name} ===")
            fn(config, ckpt_mgr)

    if not (config.hardware.distributed and dist.is_initialized()) or dist.get_rank() == 0:
        logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MyeloMemory — epigenetic drug resistance memory prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/h100.yaml"),
        help="Path to YAML config file (default: configs/h100.yaml)",
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        choices=[name for name, _ in STAGES],
        help="Run a single stage instead of the full pipeline",
    )
    parser.add_argument(
        "--resume-from-latest", action="store_true",
        help="Automatically resume from the latest checkpoint",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume a specific stage from this checkpoint file",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override API server port (only for --stage serve)",
    )
    return parser.parse_args()


def main() -> None:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    config = load_config(args.config)

    if args.port is not None:
        config.api.port = args.port

    if args.resume is not None:
        config.resume_checkpoint = args.resume

    run_pipeline(config, stage=args.stage)


if __name__ == "__main__":
    main()