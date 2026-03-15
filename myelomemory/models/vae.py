"""Module 1: Proteome-to-Epigenome Conditional Variational Autoencoder.

Given a proteomic profile (protein abundances), infers the latent epigenetic
memory state and reconstructs the expected epigenomic profile (ATAC-seq peaks,
histone modifications).

Architecture:
    Encoder: proteomics → [hidden layers] → (mu, log_var) in R^latent_dim
    Decoder: z ~ N(mu, var) → [hidden layers] → reconstructed epigenomics

Key design choices:
    - GELU activation (smoother gradients than ReLU for biological data)
    - Cyclical KL annealing (prevents posterior collapse)
    - Gradient checkpointing (fits batch_size=512 on H100 80GB)
    - bf16 mixed precision throughout
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from myelomemory.config import VAEConfig
from myelomemory.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class _EncoderBlock(nn.Module):
    """Single encoder layer: Linear → BatchNorm → GELU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_bn: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))


class _DecoderBlock(nn.Module):
    """Single decoder layer: Linear → BatchNorm → GELU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_bn: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))


class ProteomeToEpigenomeVAE(nn.Module):
    """Conditional VAE mapping proteomics → latent memory state → epigenomics.

    The latent space (64-dim by default) represents the inferred epigenetic
    memory state: a compressed representation of the chromatin landscape
    that would produce the observed proteomic profile.

    Args:
        config: VAEConfig with architecture and training hyperparameters.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim

        # Build encoder
        encoder_layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.encoder_hidden_dims:
            encoder_layers.append(
                _EncoderBlock(prev_dim, hidden_dim, config.dropout, config.use_batch_norm)
            )
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent projections
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, config.latent_dim)

        # Build decoder
        decoder_layers = []
        prev_dim = config.latent_dim
        for hidden_dim in config.decoder_hidden_dims:
            decoder_layers.append(
                _DecoderBlock(prev_dim, hidden_dim, config.dropout, config.use_batch_norm)
            )
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Output heads for each epigenomic assay
        self.output_head = nn.Linear(prev_dim, config.epigenome_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode proteomics to latent distribution parameters.

        Args:
            x: (B, P) protein abundance tensor.

        Returns:
            Tuple of (mu, log_var), each (B, latent_dim).
        """
        if self.config.gradient_checkpointing and self.training:
            h = torch.utils.checkpoint.checkpoint(
                self.encoder, x, use_reentrant=False
            )
        else:
            h = self.encoder(x)

        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std.

        Args:
            mu: (B, latent_dim) mean of the approximate posterior.
            log_var: (B, latent_dim) log variance.

        Returns:
            z: (B, latent_dim) sampled latent vector.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic at inference

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed epigenomics.

        Args:
            z: (B, latent_dim) latent memory state vector.

        Returns:
            (B, epigenome_dim) reconstructed epigenomic profile.
        """
        h = self.decoder(z)
        return self.output_head(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → sample → decode.

        Args:
            x: (B, P) protein abundance tensor.

        Returns:
            Tuple of (reconstruction, mu, log_var).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def get_memory_state(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the latent memory state (mu) without sampling.

        This is the primary output used by downstream modules.

        Args:
            x: (B, P) protein abundance tensor.

        Returns:
            (B, latent_dim) deterministic memory state embedding.
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


def _kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL divergence from N(mu, var) to N(0, I).

    Returns:
        Scalar KL loss (mean over batch).
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def _cyclical_kl_weight(
    step: int, total_steps: int, n_cycles: int, ratio: float, max_weight: float
) -> float:
    """Compute cyclical KL annealing weight.

    Follows the cyclical annealing schedule from Fu et al. (2019).

    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        n_cycles: Number of annealing cycles.
        ratio: Fraction of each cycle spent increasing weight.
        max_weight: Maximum KL weight.

    Returns:
        KL weight for the current step.
    """
    cycle_length = total_steps // n_cycles
    position_in_cycle = step % cycle_length
    ramp_length = int(cycle_length * ratio)

    if position_in_cycle < ramp_length:
        return max_weight * (position_in_cycle / ramp_length)
    return max_weight


def train_vae(
    model: nn.Module,
    dataset: Any,
    splits: dict[str, list[int]],
    config: VAEConfig,
    subset: str,
    ckpt_mgr: CheckpointManager,
    stage_name: str,
) -> dict[str, Any]:
    """Train the VAE on specified data subset.

    Args:
        model: ProteomeToEpigenomeVAE (possibly DDP-wrapped).
        dataset: MultiOmicsDataset.
        splits: Train/val/test index splits.
        config: VAE hyperparameters.
        subset: 'pan_cancer' or 'hematological'.
        ckpt_mgr: Checkpoint manager for saving.
        stage_name: Name for the checkpoint file.

    Returns:
        Dict with 'checkpoint_path' and 'metrics'.
    """
    # Select subset
    if subset == "hematological":
        heme_lineages = {
            "Myeloid", "Lymphoid",
            "haematopoietic_and_lymphoid_tissue",
        }
        indices = [
            i for i in splits["train"]
            if dataset.lineage[i] in heme_lineages
        ]
        val_indices = [
            i for i in splits["val"]
            if dataset.lineage[i] in heme_lineages
        ]
    else:
        indices = splits["train"]
        val_indices = splits["val"]

    epochs = config.pretrain_epochs if subset == "pan_cancer" else config.finetune_epochs
    lr = config.pretrain_lr if subset == "pan_cancer" else config.finetune_lr

    effective_batch_size = min(config.batch_size, len(indices))
    train_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=min(8, 2 if len(indices) < 100 else 8),
        pin_memory=True,
        drop_last=len(indices) > effective_batch_size,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda")

    total_steps = epochs * max(len(train_loader), 1)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    kl_weight = 0.0

    device = next(model.parameters()).device

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for batch in train_loader:
            proteomics = batch["proteomics"].to(device, non_blocking=True)
            epigenomics = batch["epigenomics"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):
                recon, mu, log_var = model(proteomics)
                recon_loss = F.mse_loss(recon, epigenomics)
                kl_loss = _kl_divergence(mu, log_var)

                kl_weight = _cyclical_kl_weight(
                    global_step, total_steps,
                    config.kl_anneal_cycles,
                    config.kl_anneal_ratio,
                    config.kl_weight_max,
                )
                loss = recon_loss + kl_weight * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            global_step += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                proteomics = batch["proteomics"].to(device, non_blocking=True)
                epigenomics = batch["epigenomics"].to(device, non_blocking=True)

                with autocast("cuda", dtype=torch.bfloat16):
                    recon, mu, log_var = model(proteomics)
                    recon_loss = F.mse_loss(recon, epigenomics)
                    kl_loss = _kl_divergence(mu, log_var)
                    val_loss += (recon_loss + kl_loss).item()

        avg_train = epoch_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        logger.info(
            f"[{stage_name}] Epoch {epoch + 1}/{epochs} — "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"recon={epoch_recon / max(len(train_loader), 1):.4f} "
            f"kl={epoch_kl / max(len(train_loader), 1):.4f} "
            f"kl_weight={kl_weight:.4f}"
        )

        # Early stopping + checkpoint best
        if avg_val < best_val_loss - config.min_delta:
            best_val_loss = avg_val
            patience_counter = 0

            # Save best checkpoint
            raw_model = model.module if hasattr(model, "module") else model
            ckpt_mgr.save(stage_name, {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_val_loss,
            })
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    metrics = {
        "best_val_loss": best_val_loss,
        "final_epoch": epoch + 1,
        "total_steps": global_step,
    }

    return {
        "checkpoint_path": ckpt_mgr.path(stage_name),
        "metrics": metrics,
    }