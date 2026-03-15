"""Module 2: Memory Stability Scorer — ODE-based bistability model.

Adapts the Sneppen & Ringrose chromatin bistability framework:
    - Parameterizes feedback loop strengths from REAL proteomic measurements
      of chromatin reader/writer enzymes (EZH2, DNMT1, TET1, etc.)
    - Computes basin-of-attraction depth as a stability score
    - Score ranges from 0 (transient adaptation) to 1 (locked-in memory)

The ODE system models two competing chromatin states (active vs. repressed)
with auto-catalytic and cross-inhibitory feedback. The depth of the potential
well at the current state determines how much perturbation (drug treatment)
would be needed to flip the epigenetic state.

Uses torchdiffeq for GPU-accelerated, differentiable ODE solving (H100-optimized).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from myelomemory.config import StabilityConfig
from myelomemory.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

# Lazy import — torchdiffeq is only needed for this module
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


class ChromatinODE(nn.Module):
    """ODE system for the Sneppen-Ringrose bistability model.

    State variables:
        a: Active chromatin mark level (e.g., H3K4me3)
        r: Repressive chromatin mark level (e.g., H3K27me3)

    Dynamics:
        da/dt = w_a * f(a) - e_r * g(r) * a - d_a * a + basal_a
        dr/dt = w_r * f(r) - e_a * g(a) * r - d_r * r + basal_r

    Where:
        w_a, w_r: Writer strengths (auto-catalysis, from proteomic levels)
        e_a, e_r: Eraser strengths (cross-inhibition, from proteomic levels)
        d_a, d_r: Dilution rates (from proliferation markers)
        f, g: Hill functions for cooperative binding
        basal_a, basal_r: Basal production rates

    Parameters are derived from chromatin reader/writer protein abundances.
    """

    def __init__(self, config: StabilityConfig) -> None:
        super().__init__()
        n_proteins = len(config.reader_writer_proteins)

        # Learnable mapping: protein levels → ODE parameters
        # This is calibrated against washout time-course data
        self.protein_to_params = nn.Sequential(
            nn.Linear(n_proteins, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 8),  # [w_a, w_r, e_a, e_r, d_a, d_r, basal_a, basal_r]
            nn.Softplus(),  # All ODE params must be positive
        )

        # Hill function parameters (learnable)
        self.hill_n = nn.Parameter(torch.tensor(2.0))  # Cooperativity
        self.hill_k = nn.Parameter(torch.tensor(0.5))  # Half-max

    def _hill(self, x: torch.Tensor) -> torch.Tensor:
        """Hill function for cooperative binding."""
        n = F.softplus(self.hill_n)  # Ensure n > 0
        k = F.softplus(self.hill_k)
        return x.pow(n) / (k.pow(n) + x.pow(n) + 1e-8)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute derivatives for the chromatin ODE system.

        Args:
            t: Current time (scalar, unused but required by odeint).
            state: (B, 2 + 8) tensor where [:, 0] = a, [:, 1] = r,
                   [:, 2:] = ODE parameters (constant through integration).

        Returns:
            (B, 2 + 8) derivatives (params have zero derivative).
        """
        a = state[:, 0:1]  # Active mark level
        r = state[:, 1:2]  # Repressive mark level
        params = state[:, 2:]  # ODE parameters (constant)

        w_a = params[:, 0:1]
        w_r = params[:, 1:2]
        e_a = params[:, 2:3]
        e_r = params[:, 3:4]
        d_a = params[:, 4:5]
        d_r = params[:, 5:6]
        basal_a = params[:, 6:7]
        basal_r = params[:, 7:8]

        da_dt = w_a * self._hill(a) - e_r * self._hill(r) * a - d_a * a + basal_a
        dr_dt = w_r * self._hill(r) - e_a * self._hill(a) * r - d_r * r + basal_r

        # Parameters are constant — zero derivatives
        dparam_dt = torch.zeros_like(params)

        return torch.cat([da_dt, dr_dt, dparam_dt], dim=1)


# Use F from torch.nn.functional for Hill function
import torch.nn.functional as F


class MemoryStabilityScorer(nn.Module):
    """Computes the memory stability score for a given proteomic profile.

    Pipeline:
        1. Extract chromatin reader/writer protein levels from full proteome
        2. Map protein levels → ODE parameters via learned neural network
        3. Integrate ODE to find steady state
        4. Estimate basin-of-attraction depth via perturbation sampling
        5. Normalize to 0–1 stability score

    A score of 0 means the epigenetic state is easily flipped (transient
    adaptation, potentially reversible by drug rechallenge).

    A score of 1 means the state is deeply locked in (permanent epigenetic
    memory, resistant to perturbation).

    Args:
        config: StabilityConfig with ODE and calibration parameters.
    """

    def __init__(self, config: StabilityConfig) -> None:
        super().__init__()
        self.config = config
        self.ode = ChromatinODE(config)
        self.protein_names = config.reader_writer_proteins

        # Learnable scale/bias for final score normalization
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.tensor(0.0))

    def extract_reader_writer_levels(
        self,
        proteomics: torch.Tensor,
        all_protein_names: list[str],
    ) -> torch.Tensor:
        """Extract chromatin reader/writer protein levels from full proteome.

        Args:
            proteomics: (B, P) full protein abundance tensor.
            all_protein_names: List of P protein names matching columns.

        Returns:
            (B, N_rw) tensor of reader/writer protein levels.
        """
        name_to_idx = {name: i for i, name in enumerate(all_protein_names)}
        indices = []
        for prot in self.protein_names:
            if prot in name_to_idx:
                indices.append(name_to_idx[prot])
            else:
                # Use zero for missing proteins
                indices.append(-1)

        result = torch.zeros(
            proteomics.shape[0], len(self.protein_names),
            device=proteomics.device, dtype=proteomics.dtype,
        )
        for i, idx in enumerate(indices):
            if idx >= 0:
                result[:, i] = proteomics[:, idx]

        return result

    def _find_steady_state(
        self, ode_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate ODE to find the steady-state chromatin configuration.

        Args:
            ode_params: (B, 8) ODE parameters from protein_to_params network.

        Returns:
            Tuple of (a_steady, r_steady), each (B, 1).
        """
        if odeint is None:
            raise ImportError(
                "torchdiffeq is required for stability scoring. "
                "Install with: pip install torchdiffeq"
            )

        batch_size = ode_params.shape[0]
        device = ode_params.device

        # Initial condition: balanced state
        a0 = torch.full((batch_size, 1), 0.5, device=device)
        r0 = torch.full((batch_size, 1), 0.5, device=device)
        state0 = torch.cat([a0, r0, ode_params], dim=1)

        t_span = torch.tensor(
            [0.0, self.config.integration_time], device=device
        )

        # Integrate
        trajectory = odeint(
            self.ode,
            state0,
            t_span,
            method=self.config.ode_solver,
            rtol=self.config.ode_rtol,
            atol=self.config.ode_atol,
        )

        final_state = trajectory[-1]  # (B, 10)
        a_steady = final_state[:, 0:1]
        r_steady = final_state[:, 1:2]

        return a_steady, r_steady

    def _estimate_basin_depth(
        self,
        ode_params: torch.Tensor,
        a_steady: torch.Tensor,
        r_steady: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate basin-of-attraction depth via Monte Carlo perturbation.

        Perturbs the steady state and measures how many perturbations
        return to the same attractor vs. flip to the other.

        The fraction that return = proxy for basin depth = stability.

        Args:
            ode_params: (B, 8) ODE parameters.
            a_steady: (B, 1) steady-state active mark level.
            r_steady: (B, 1) steady-state repressive mark level.

        Returns:
            (B,) basin depth scores (higher = more stable).
        """
        batch_size = ode_params.shape[0]
        device = ode_params.device
        n_samples = self.config.n_perturbation_samples

        # Generate random perturbations
        perturbations = torch.randn(
            n_samples, batch_size, 2, device=device
        ) * 0.3  # Scale of perturbation

        return_count = torch.zeros(batch_size, device=device)

        steady = torch.cat([a_steady, r_steady], dim=1)  # (B, 2)

        for i in range(n_samples):
            perturbed = (steady + perturbations[i]).clamp(min=0.01)
            perturbed_state = torch.cat([perturbed, ode_params], dim=1)

            t_span = torch.tensor(
                [0.0, self.config.integration_time], device=device
            )

            with torch.no_grad():
                result = odeint(
                    self.ode,
                    perturbed_state,
                    t_span,
                    method="euler",  # Faster for perturbation sampling
                    options={"step_size": 1.0},
                )

            final = result[-1, :, :2]  # (B, 2)
            # Check if returned to same attractor (within tolerance)
            distance = (final - steady).norm(dim=1)
            returned = (distance < 0.2).float()
            return_count += returned

        basin_depth = return_count / n_samples  # Fraction that returned
        return basin_depth

    def forward(
        self,
        proteomics: torch.Tensor,
        all_protein_names: list[str],
    ) -> torch.Tensor:
        """Compute memory stability score for a batch of proteomic profiles.

        Args:
            proteomics: (B, P) protein abundance tensor.
            all_protein_names: List of P protein names.

        Returns:
            (B,) stability scores in [0, 1].
        """
        # Step 1: Extract reader/writer levels
        rw_levels = self.extract_reader_writer_levels(proteomics, all_protein_names)

        # Step 2: Map to ODE parameters
        ode_params = self.ode.protein_to_params(rw_levels)

        # Step 3: Find steady state
        a_steady, r_steady = self._find_steady_state(ode_params)

        # Step 4: Estimate basin depth
        basin_depth = self._estimate_basin_depth(ode_params, a_steady, r_steady)

        # Step 5: Normalize to [0, 1]
        score = torch.sigmoid(self.score_scale * basin_depth + self.score_bias)

        return score


def calibrate_scorer(
    scorer: MemoryStabilityScorer,
    dataset: Any,
    vae_checkpoint: dict[str, Any],
    config: StabilityConfig,
    ckpt_mgr: CheckpointManager,
) -> dict[str, Any]:
    """Calibrate the stability scorer against drug washout time-course data.

    The calibration objective: cell lines that show persistent drug resistance
    after washout should have HIGH stability scores; those that revert should
    have LOW stability scores.

    Since direct washout data may be limited, we use a proxy: the variance
    of drug sensitivity across similar cell lines. High variance = low
    stability (the state is noisy/unstable). Low variance = high stability
    (the state is consistent/locked).

    Args:
        scorer: MemoryStabilityScorer model.
        dataset: MultiOmicsDataset.
        vae_checkpoint: Loaded VAE checkpoint (for memory state extraction).
        config: StabilityConfig.
        ckpt_mgr: Checkpoint manager.

    Returns:
        Dict with 'checkpoint_path' and 'metrics'.
    """
    device = next(scorer.parameters()).device

    optimizer = torch.optim.Adam(
        scorer.parameters(), lr=config.calibration_lr
    )

    # Use drug sensitivity variance as proxy for instability
    drug_sens = dataset.drug_sensitivity  # (N, D)
    # Compute per-sample variance across drugs (ignoring NaN)
    drug_var = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        valid = drug_sens[i][~torch.isnan(drug_sens[i])]
        if len(valid) > 1:
            drug_var[i] = valid.var().item()
        else:
            drug_var[i] = float("nan")

    # Normalize variance to [0, 1] target (high var → low stability target)
    valid_mask = ~torch.isnan(drug_var)
    if valid_mask.sum() > 0:
        dv = drug_var[valid_mask]
        drug_var_norm = torch.zeros_like(drug_var)
        drug_var_norm[valid_mask] = 1.0 - (dv - dv.min()) / (dv.max() - dv.min() + 1e-8)
    else:
        logger.warning("No valid drug sensitivity data for calibration; using uniform targets")
        drug_var_norm = torch.full((len(dataset),), 0.5)
        valid_mask = torch.ones(len(dataset), dtype=torch.bool)

    # Calibration loop
    best_loss = float("inf")
    protein_names = dataset.protein_names

    for epoch in range(config.calibration_epochs):
        # Mini-batch from valid samples
        valid_indices = torch.where(valid_mask)[0]
        perm = valid_indices[torch.randperm(len(valid_indices))]
        batch_idx = perm[:config.calibration_batch_size]

        proteomics = dataset.proteomics[batch_idx].to(device)
        targets = drug_var_norm[batch_idx].to(device)

        optimizer.zero_grad(set_to_none=True)

        scores = scorer(proteomics, protein_names)
        loss = F.mse_loss(scores, targets)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            logger.info(
                f"[stability_calibrate] Epoch {epoch + 1}/{config.calibration_epochs} "
                f"loss={loss.item():.4f}"
            )

        if loss.item() < best_loss:
            best_loss = loss.item()
            ckpt_mgr.save("stability_calibrated", {
                "model_state_dict": scorer.state_dict(),
                "epoch": epoch,
                "best_metric": best_loss,
            })

    metrics = {
        "best_calibration_loss": best_loss,
        "n_valid_samples": int(valid_mask.sum()),
    }

    return {
        "checkpoint_path": ckpt_mgr.path("stability_calibrated"),
        "metrics": metrics,
    }
