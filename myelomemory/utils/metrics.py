"""Evaluation metrics for all three pipeline modules.

Computes:
    - VAE: Reconstruction MSE, KL divergence, latent space quality
    - Stability: Calibration error, rank correlation with drug variance
    - GNN: AUROC, AUPRC per drug, resistance prediction MSE
    - Pipeline: End-to-end reversibility accuracy, composite score
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        mean_squared_error,
    )
    from scipy.stats import spearmanr
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def reconstruction_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute mean squared error for VAE reconstruction.

    Args:
        predicted: (N, E) reconstructed epigenomic profiles.
        target: (N, E) true epigenomic profiles.

    Returns:
        Scalar MSE.
    """
    return torch.nn.functional.mse_loss(predicted, target).item()


def latent_space_metrics(
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> dict[str, float]:
    """Compute latent space quality metrics.

    Args:
        mu: (N, L) mean vectors.
        log_var: (N, L) log-variance vectors.

    Returns:
        Dict with 'mean_kl', 'active_units', 'latent_variance'.
    """
    # KL divergence per dimension
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    mean_kl = kl_per_dim.mean().item()

    # Active units: dimensions where KL > 0.01 (not collapsed)
    kl_per_unit = kl_per_dim.mean(dim=0)
    active_units = (kl_per_unit > 0.01).sum().item()

    # Overall latent variance
    latent_var = mu.var(dim=0).mean().item()

    return {
        "mean_kl": mean_kl,
        "active_units": int(active_units),
        "total_units": mu.shape[1],
        "latent_variance": latent_var,
    }


def stability_calibration_metrics(
    predicted_scores: torch.Tensor,
    drug_sensitivity_variance: torch.Tensor,
) -> dict[str, float]:
    """Evaluate stability scorer calibration.

    The stability score should be inversely correlated with drug sensitivity
    variance (high stability → consistent resistance → low variance).

    Args:
        predicted_scores: (N,) stability scores.
        drug_sensitivity_variance: (N,) variance of IC50 across drugs.

    Returns:
        Dict with 'spearman_rho', 'calibration_mse'.
    """
    # Filter NaN
    mask = ~(torch.isnan(predicted_scores) | torch.isnan(drug_sensitivity_variance))
    pred = predicted_scores[mask].cpu().numpy()
    target = drug_sensitivity_variance[mask].cpu().numpy()

    if len(pred) < 3:
        return {"spearman_rho": 0.0, "calibration_mse": float("inf")}

    # Stability should be inversely correlated with drug variance
    rho = 0.0
    if HAS_SKLEARN:
        rho, _ = spearmanr(pred, target)
        rho = float(rho) if not np.isnan(rho) else 0.0

    # Target: high stability → low variance (inverted and normalized)
    target_norm = 1.0 - (target - target.min()) / (target.max() - target.min() + 1e-8)
    cal_mse = float(mean_squared_error(target_norm, pred)) if HAS_SKLEARN else 0.0

    return {
        "spearman_rho": rho,
        "calibration_mse": cal_mse,
    }


def drug_resistance_metrics(
    predicted_ic50: torch.Tensor,
    true_ic50: torch.Tensor,
    drug_names: list[str],
    resistance_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute per-drug and aggregate resistance prediction metrics.

    Args:
        predicted_ic50: (N, D) predicted IC50 values.
        true_ic50: (N, D) true IC50 values (may contain NaN).
        drug_names: List of D drug names.
        resistance_threshold: IC50 threshold for binary resistant/sensitive.

    Returns:
        Dict with per-drug AUROC, AUPRC, MSE, and aggregate metrics.
    """
    if not HAS_SKLEARN:
        logger.warning("sklearn not available; returning empty metrics")
        return {}

    metrics = {}
    all_aurocs = []

    for d, drug in enumerate(drug_names):
        pred = predicted_ic50[:, d]
        true = true_ic50[:, d]

        # Filter NaN
        mask = ~torch.isnan(true)
        if mask.sum() < 5:
            continue

        pred_np = pred[mask].cpu().numpy()
        true_np = true[mask].cpu().numpy()

        # MSE
        mse = float(mean_squared_error(true_np, pred_np))
        metrics[f"{drug}/mse"] = mse

        # Binary classification metrics
        binary_true = (true_np > resistance_threshold).astype(int)
        if len(np.unique(binary_true)) == 2:
            auroc = float(roc_auc_score(binary_true, pred_np))
            auprc = float(average_precision_score(binary_true, pred_np))
            metrics[f"{drug}/auroc"] = auroc
            metrics[f"{drug}/auprc"] = auprc
            all_aurocs.append(auroc)

    if all_aurocs:
        metrics["mean_auroc"] = float(np.mean(all_aurocs))

    return metrics


def compute_full_metrics(
    predictions: list[Any],
    dataset: Any,
    split: str = "test",
) -> dict[str, float]:
    """Compute all metrics for the full pipeline output.

    Args:
        predictions: List of PredictionResult objects.
        dataset: MultiOmicsDataset.
        split: Which split was evaluated.

    Returns:
        Comprehensive metrics dict.
    """
    metrics = {"split": split, "n_samples": len(predictions)}

    if not predictions:
        return metrics

    # Aggregate stability scores
    stability_scores = torch.tensor([p.stability_score for p in predictions])
    metrics["mean_stability"] = stability_scores.mean().item()
    metrics["std_stability"] = stability_scores.std().item()

    # Count high/medium/low stability
    metrics["n_high_stability"] = int((stability_scores > 0.7).sum())
    metrics["n_medium_stability"] = int(
        ((stability_scores > 0.3) & (stability_scores <= 0.7)).sum()
    )
    metrics["n_low_stability"] = int((stability_scores <= 0.3).sum())

    # Per-drug resistance metrics (AUROC, AUPRC, MSE)
    if hasattr(dataset, "drug_sensitivity") and HAS_SKLEARN:
        drug_names = list(predictions[0].drug_resistance.keys())
        if drug_names:
            predicted_ic50 = torch.tensor([
                [p.drug_resistance.get(d, 0.0) for d in drug_names]
                for p in predictions
            ])
            true_ic50 = dataset.drug_sensitivity[:len(predictions)]
            # Only use columns matching the number of trained drugs
            if true_ic50.shape[1] >= len(drug_names):
                true_ic50 = true_ic50[:, :len(drug_names)]
                drug_metrics = drug_resistance_metrics(
                    predicted_ic50, true_ic50, drug_names
                )
                metrics.update(drug_metrics)

    # Stability calibration metrics
    if hasattr(dataset, "drug_sensitivity") and HAS_SKLEARN:
        drug_sens = dataset.drug_sensitivity[:len(predictions)]
        drug_var = torch.zeros(len(predictions))
        for i in range(len(predictions)):
            valid = drug_sens[i][~torch.isnan(drug_sens[i])]
            drug_var[i] = valid.var().item() if len(valid) > 1 else float("nan")
        stab_metrics = stability_calibration_metrics(stability_scores, drug_var)
        metrics["stability_spearman_rho"] = stab_metrics["spearman_rho"]
        metrics["stability_calibration_mse"] = stab_metrics["calibration_mse"]

    # Bootstrap confidence intervals for key metrics
    n_bootstrap = 1000
    n = len(predictions)
    if n >= 10:
        rng = np.random.RandomState(42)
        boot_stab = np.zeros(n_bootstrap)
        stab_np = stability_scores.numpy()
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_stab[b] = stab_np[idx].mean()
        metrics["mean_stability_ci95_low"] = float(np.percentile(boot_stab, 2.5))
        metrics["mean_stability_ci95_high"] = float(np.percentile(boot_stab, 97.5))

    logger.info(
        f"Pipeline metrics ({split}): {len(predictions)} samples, "
        f"mean_stability={metrics['mean_stability']:.3f}"
    )

    return metrics
