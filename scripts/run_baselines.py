#!/usr/bin/env python3
"""Baseline comparison script for the MyeloMemory pipeline.

Implements 5 baselines to demonstrate that the full VAE + ODE + GNN architecture
adds value over simpler approaches:

    a) RandomForest on raw proteomics -> drug IC50
    b) ElasticNet on raw proteomics -> drug IC50
    c) GNN without memory features (zero out memory_state and stability_score)
    d) GNN without stability score (keep memory, set stability = 0.5)
    e) Variance-based heuristic (rank by drug sensitivity variance)

Usage:
    python scripts/run_baselines.py --checkpoint-dir checkpoints --output results/baseline_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    mask = ~np.isnan(target)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((pred[mask] - target[mask]) ** 2))


def compute_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = ~np.isnan(target)
    if mask.sum() < 3:
        return float("nan")
    corr, _ = spearmanr(pred[mask], target[mask])
    return float(corr)


def compute_auroc(pred: np.ndarray, target: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    mask = ~np.isnan(target)
    if mask.sum() < 5:
        return float("nan")
    t, p = target[mask], pred[mask]
    labels = (t >= float(np.median(t))).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    try:
        return float(roc_auc_score(labels, p))
    except ValueError:
        return float("nan")


def evaluate_predictions(pred: np.ndarray, target: np.ndarray, drug_names: list[str]) -> dict:
    results: dict[str, Any] = {"per_drug": {}, "aggregate": {}}
    all_mse, all_rho, all_auroc = [], [], []
    for d, name in enumerate(drug_names):
        mse = compute_mse(pred[:, d], target[:, d])
        rho = compute_spearman(pred[:, d], target[:, d])
        auroc = compute_auroc(pred[:, d], target[:, d])
        results["per_drug"][name] = {"mse": mse, "spearman_rho": rho, "auroc": auroc}
        if not np.isnan(mse): all_mse.append(mse)
        if not np.isnan(rho): all_rho.append(rho)
        if not np.isnan(auroc): all_auroc.append(auroc)
    results["aggregate"] = {
        "mean_mse": float(np.mean(all_mse)) if all_mse else float("nan"),
        "mean_spearman_rho": float(np.mean(all_rho)) if all_rho else float("nan"),
        "mean_auroc": float(np.mean(all_auroc)) if all_auroc else float("nan"),
    }
    return results


def run_random_forest(X_train, y_train, X_test, y_test, drug_names):
    from sklearn.ensemble import RandomForestRegressor
    logger.info("Running baseline: RandomForest")
    pred = np.full_like(y_test, np.nan, dtype=np.float64)
    for d in range(y_train.shape[1]):
        mask = ~np.isnan(y_train[:, d])
        if mask.sum() < 5: continue
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
        rf.fit(X_train[mask], y_train[mask, d])
        pred[:, d] = rf.predict(X_test)
    return {"name": "RandomForest", "metrics": evaluate_predictions(pred, y_test, drug_names)}


def run_elastic_net(X_train, y_train, X_test, y_test, drug_names):
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    logger.info("Running baseline: ElasticNet")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    pred = np.full_like(y_test, np.nan, dtype=np.float64)
    for d in range(y_train.shape[1]):
        mask = ~np.isnan(y_train[:, d])
        if mask.sum() < 5: continue
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=5, max_iter=5000, random_state=42, n_jobs=-1)
        en.fit(Xtr[mask], y_train[mask, d])
        pred[:, d] = en.predict(Xte)
    return {"name": "ElasticNet", "metrics": evaluate_predictions(pred, y_test, drug_names)}


def _build_gnn_batch(proteomics, memory_states, stability_scores, edge_index, edge_attr, device):
    from torch_geometric.data import Data as PyGData
    B, P = proteomics.shape
    E = edge_index.shape[1]
    node_x = torch.cat([
        proteomics.unsqueeze(2),
        memory_states.unsqueeze(1).expand(-1, P, -1),
        stability_scores.unsqueeze(1).unsqueeze(2).expand(-1, P, 1),
    ], dim=2).reshape(B * P, -1)
    offsets = torch.arange(B, device=device).unsqueeze(1) * P
    batch_ei = (edge_index.unsqueeze(0).expand(B, -1, -1) + offsets.unsqueeze(1)).reshape(2, B * E)
    batch_ea = edge_attr.unsqueeze(0).expand(B, -1, -1).reshape(B * E, -1)
    batch_vec = torch.arange(B, device=device).unsqueeze(1).expand(-1, P).reshape(-1)
    data = PyGData(x=node_x, edge_index=batch_ei, edge_attr=batch_ea)
    data.batch = batch_vec
    data.num_graphs = B
    return data


def _load_gnn(checkpoint_dir, device):
    from myelomemory.config import GNNConfig
    from myelomemory.models.gnn import ResistanceGNN
    gnn_ckpt = torch.load(checkpoint_dir / "gnn_trained.pt", map_location=device, weights_only=False)
    sd = gnn_ckpt["model_state_dict"]
    cfg = GNNConfig(hidden_dim=128, num_layers=3, num_heads=4, node_feature_dim=66, num_drugs=2)
    if "node_proj.0.weight" in sd:
        cfg.node_feature_dim = sd["node_proj.0.weight"].shape[1]
        cfg.hidden_dim = sd["node_proj.0.weight"].shape[0]
    if "resistance_head.3.weight" in sd:
        cfg.num_drugs = sd["resistance_head.3.weight"].shape[0]
    idx = 0
    while f"conv_layers.{idx}.bias" in sd: idx += 1
    if idx > 0: cfg.num_layers = idx
    gnn = ResistanceGNN(cfg).to(device)
    gnn.load_state_dict(sd)
    gnn.eval()
    return gnn


def _load_ppi(dataset, device):
    from myelomemory.data.graph_builder import build_protein_index, build_edge_index
    protein_index = build_protein_index(dataset.protein_names)
    if hasattr(dataset, "ppi_edges") and dataset.ppi_edges:
        ei, ea = build_edge_index(dataset.ppi_edges, dataset.ppi_scores, protein_index)
    else:
        n = len(dataset.protein_names)
        ei = torch.stack([torch.arange(n - 1), torch.arange(1, n)])
        ea = torch.ones(n - 1, 1)
    return ei.to(device), ea.to(device)


def _get_latent_dim(checkpoint_dir, device):
    sd = torch.load(checkpoint_dir / "vae_finetuned.pt", map_location=device, weights_only=False)["model_state_dict"]
    return sd["mu_layer.weight"].shape[0] if "mu_layer.weight" in sd else 64


def run_gnn_no_memory(dataset, splits, checkpoint_dir, drug_names):
    logger.info("Running baseline: GNN without memory features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = _load_gnn(checkpoint_dir, device)
    ei, ea = _load_ppi(dataset, device)
    latent_dim = _get_latent_dim(checkpoint_dir, device)
    test_idx = splits["test"]
    prot = dataset.proteomics[test_idx].to(device)
    y_test = dataset.drug_sensitivity[test_idx].numpy()
    B, P = prot.shape
    mem_zeros = torch.zeros(B, latent_dim, device=device)
    stab_zeros = torch.zeros(B, device=device)
    preds = []
    with torch.no_grad():
        for s in range(0, B, 32):
            e = min(s + 32, B)
            batch = _build_gnn_batch(prot[s:e], mem_zeros[s:e], stab_zeros[s:e], ei, ea, device)
            preds.append(gnn(batch)["resistance"].cpu().numpy())
    pred = np.concatenate(preds)
    return {"name": "GNN_no_memory", "metrics": evaluate_predictions(pred, y_test, drug_names)}


def run_gnn_no_stability(dataset, splits, checkpoint_dir, drug_names):
    logger.info("Running baseline: GNN without stability score")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = _load_gnn(checkpoint_dir, device)
    ei, ea = _load_ppi(dataset, device)

    from myelomemory.config import VAEConfig
    from myelomemory.models.vae import ProteomeToEpigenomeVAE
    vae_ckpt = torch.load(checkpoint_dir / "vae_finetuned.pt", map_location=device, weights_only=False)
    vae_sd = vae_ckpt["model_state_dict"]
    vae_cfg = VAEConfig()
    if "encoder.0.linear.weight" in vae_sd: vae_cfg.input_dim = vae_sd["encoder.0.linear.weight"].shape[1]
    if "output_head.weight" in vae_sd: vae_cfg.epigenome_dim = vae_sd["output_head.weight"].shape[0]
    if "mu_layer.weight" in vae_sd: vae_cfg.latent_dim = vae_sd["mu_layer.weight"].shape[0]
    vae = ProteomeToEpigenomeVAE(vae_cfg).to(device)
    vae.load_state_dict(vae_sd)
    vae.eval()

    test_idx = splits["test"]
    prot = dataset.proteomics[test_idx].to(device)
    y_test = dataset.drug_sensitivity[test_idx].numpy()
    B, P = prot.shape

    all_mem = []
    with torch.no_grad():
        for s in range(0, B, 128):
            all_mem.append(vae.get_memory_state(prot[s:min(s+128, B)]))
    all_mem = torch.cat(all_mem)
    stab_const = torch.full((B,), 0.5, device=device)

    preds = []
    with torch.no_grad():
        for s in range(0, B, 32):
            e = min(s + 32, B)
            batch = _build_gnn_batch(prot[s:e], all_mem[s:e], stab_const[s:e], ei, ea, device)
            preds.append(gnn(batch)["resistance"].cpu().numpy())
    pred = np.concatenate(preds)
    return {"name": "GNN_no_stability", "metrics": evaluate_predictions(pred, y_test, drug_names)}


def run_variance_heuristic(y_train, y_test, drug_names):
    logger.info("Running baseline: Variance heuristic")
    pred = np.full_like(y_test, np.nan, dtype=np.float64)
    for d in range(y_train.shape[1]):
        mask = ~np.isnan(y_train[:, d])
        if mask.sum() < 2: continue
        pred[:, d] = float(np.mean(y_train[mask, d]))
    return {"name": "VarianceHeuristic", "metrics": evaluate_predictions(pred, y_test, drug_names)}


def print_table(results, drug_names):
    hdr = f"{'Model':<25} {'Mean MSE':>10} {'Mean Rho':>10} {'Mean AUROC':>11}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n  BASELINE COMPARISON\n{sep}\n{hdr}\n{sep}")
    for r in results:
        a = r["metrics"]["aggregate"]
        def f(v): return f"{v:.4f}" if not np.isnan(v) else "N/A"
        print(f"{r['name']:<25} {f(a['mean_mse']):>10} {f(a['mean_spearman_rho']):>10} {f(a['mean_auroc']):>11}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_comparison.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_ckpt = torch.load(args.checkpoint_dir / "data_ready.pt", map_location="cpu", weights_only=False)
    dataset = data_ckpt["dataset"]
    splits = data_ckpt["splits"]

    num_drugs = dataset.drug_sensitivity.shape[1]
    drug_names = ["Bortezomib", "Lenalidomide", "Dexamethasone", "Carfilzomib", "Pomalidomide", "Daratumumab"][:num_drugs]

    X_train = np.nan_to_num(dataset.proteomics[splits["train"]].numpy())
    y_train = dataset.drug_sensitivity[splits["train"]].numpy()
    X_test = np.nan_to_num(dataset.proteomics[splits["test"]].numpy())
    y_test = dataset.drug_sensitivity[splits["test"]].numpy()

    results = []
    for fn, args_tuple in [
        (run_random_forest, (X_train, y_train, X_test, y_test, drug_names)),
        (run_elastic_net, (X_train, y_train, X_test, y_test, drug_names)),
        (run_gnn_no_memory, (dataset, splits, args.checkpoint_dir, drug_names)),
        (run_gnn_no_stability, (dataset, splits, args.checkpoint_dir, drug_names)),
        (run_variance_heuristic, (y_train, y_test, drug_names)),
    ]:
        try:
            results.append(fn(*args_tuple))
        except Exception as e:
            logger.error("%s failed: %s", fn.__name__, e)

    if results:
        print_table(results, drug_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    def sanitize(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
        if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [sanitize(v) for v in obj]
        return obj

    json_out = {"metadata": {"drug_names": drug_names, "train_size": len(splits["train"]), "test_size": len(splits["test"])},
                "baselines": {r["name"]: r["metrics"] for r in results}}
    with open(args.output, "w") as f:
        json.dump(sanitize(json_out), f, indent=2)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
