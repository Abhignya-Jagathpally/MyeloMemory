"""Structured logging with optional Weights & Biases integration.

Provides consistent logging across all pipeline stages with:
    - Console output with stage-aware formatting
    - Optional W&B metric tracking
    - Stage timing for performance monitoring
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from myelomemory.config import MyeloMemoryConfig

# Module-level state for stage timing
_stage_start_times: dict[str, float] = {}

# Optional W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def setup_logger(config: MyeloMemoryConfig) -> logging.Logger:
    """Configure the root logger for the pipeline.

    Sets up console handler with structured formatting and optionally
    initializes Weights & Biases for experiment tracking.

    Args:
        config: Pipeline configuration.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("myelomemory")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # W&B initialization
    if HAS_WANDB and config.wandb_project:
        try:
            wandb.init(
                project=config.wandb_project,
                config={
                    "vae_latent_dim": config.vae.latent_dim,
                    "vae_pretrain_epochs": config.vae.pretrain_epochs,
                    "gnn_num_layers": config.gnn.num_layers,
                    "gnn_conv_type": config.gnn.conv_type,
                    "hardware_dtype": config.hardware.dtype,
                },
                reinit=True,
            )
            logger.info(f"W&B initialized: project={config.wandb_project}")
        except Exception as e:
            logger.warning(f"W&B init failed (continuing without): {e}")

    return logger


def log_stage_start(stage_name: str) -> None:
    """Log the start of a pipeline stage and begin timing.

    Args:
        stage_name: Name of the pipeline stage.
    """
    logger = logging.getLogger("myelomemory")
    _stage_start_times[stage_name] = time.time()
    logger.info(f">>> STAGE START: {stage_name}")

    if HAS_WANDB and wandb.run is not None:
        wandb.log({f"stage/{stage_name}/started": 1})


def log_stage_end(stage_name: str, metrics: dict[str, Any] | None = None) -> None:
    """Log the completion of a pipeline stage with timing and metrics.

    Args:
        stage_name: Name of the pipeline stage.
        metrics: Optional dict of metrics to log.
    """
    logger = logging.getLogger("myelomemory")

    elapsed = time.time() - _stage_start_times.get(stage_name, time.time())
    minutes = elapsed / 60

    metrics_str = ""
    if metrics:
        metrics_str = " | " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())

    logger.info(f"<<< STAGE END: {stage_name} ({minutes:.1f} min){metrics_str}")

    if HAS_WANDB and wandb.run is not None:
        log_data = {
            f"stage/{stage_name}/elapsed_minutes": minutes,
            f"stage/{stage_name}/completed": 1,
        }
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"stage/{stage_name}/{k}"] = v
        wandb.log(log_data)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to both console and W&B.

    Args:
        metrics: Dict of metric name → value.
        step: Optional global step for x-axis alignment.
    """
    logger = logging.getLogger("myelomemory")
    metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f"Metrics: {metrics_str}")

    if HAS_WANDB and wandb.run is not None:
        wandb.log(metrics, step=step)
