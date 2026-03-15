"""End-to-end inference pipeline combining all three modules.

Given a raw proteomic profile, produces:
    1. Inferred epigenetic memory state (64-dim embedding)
    2. Stability score (0 = transient, 1 = locked-in)
    3. Per-drug resistance predictions with reversibility flags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from myelomemory.config import MyeloMemoryConfig
from myelomemory.models.vae import ProteomeToEpigenomeVAE
from myelomemory.models.stability import MemoryStabilityScorer
from myelomemory.models.gnn import ResistanceGNN
from myelomemory.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for a single sample's prediction output.

    Attributes:
        memory_state: (latent_dim,) inferred epigenetic memory state vector.
        stability_score: Scalar in [0, 1]. 0=transient, 1=locked-in.
        drug_resistance: Dict mapping drug name → predicted IC50.
        drug_reversibility: Dict mapping drug name → P(reversible).
        reconstructed_epigenome: (epigenome_dim,) reconstructed epigenomic profile.
    """
    memory_state: torch.Tensor
    stability_score: float
    drug_resistance: dict[str, float]
    drug_reversibility: dict[str, float]
    reconstructed_epigenome: torch.Tensor | None = None


class MyeloMemoryPipeline:
    """Full inference pipeline wiring Modules 1–3.

    Usage:
        pipeline = MyeloMemoryPipeline.from_checkpoints(ckpt_mgr, config)
        results = pipeline.predict_single(proteomic_vector, protein_names)

    Args:
        vae: Trained ProteomeToEpigenomeVAE.
        stability_scorer: Calibrated MemoryStabilityScorer.
        gnn: Trained ResistanceGNN.
        config: MyeloMemoryConfig.
        drug_names: List of target drug names matching GNN output columns.
    """

    def __init__(
        self,
        vae: ProteomeToEpigenomeVAE,
        stability_scorer: MemoryStabilityScorer,
        gnn: ResistanceGNN,
        config: MyeloMemoryConfig,
        drug_names: list[str],
    ) -> None:
        self.vae = vae.eval()
        self.stability_scorer = stability_scorer.eval()
        self.gnn = gnn.eval()
        self.config = config
        self.drug_names = drug_names
        self.device = config.device

    @classmethod
    def from_checkpoints(
        cls,
        ckpt_mgr: CheckpointManager,
        config: MyeloMemoryConfig,
    ) -> "MyeloMemoryPipeline":
        """Load all three modules from saved checkpoints.

        Args:
            ckpt_mgr: Checkpoint manager with saved stage checkpoints.
            config: Full pipeline configuration.

        Returns:
            Initialized MyeloMemoryPipeline ready for inference.
        """
        device = config.device

        # Load VAE — infer dimensions from checkpoint state dict
        vae_ckpt = ckpt_mgr.load("vae_finetuned")
        vae_cfg = config.vae
        sd = vae_ckpt["model_state_dict"]
        if "encoder.0.linear.weight" in sd:
            vae_cfg.input_dim = sd["encoder.0.linear.weight"].shape[1]
        if "output_head.weight" in sd:
            vae_cfg.epigenome_dim = sd["output_head.weight"].shape[0]
        if "mu_layer.weight" in sd:
            vae_cfg.latent_dim = sd["mu_layer.weight"].shape[0]
        vae = ProteomeToEpigenomeVAE(vae_cfg).to(device)
        vae.load_state_dict(sd)
        logger.info("Loaded VAE from vae_finetuned checkpoint (input_dim=%d, epigenome_dim=%d)",
                     vae_cfg.input_dim, vae_cfg.epigenome_dim)

        # Load stability scorer
        stab_ckpt = ckpt_mgr.load("stability_calibrated")
        scorer = MemoryStabilityScorer(config.stability).to(device)
        scorer.load_state_dict(stab_ckpt["model_state_dict"])
        logger.info("Loaded stability scorer from stability_calibrated checkpoint")

        # Load GNN — infer all dimensions from checkpoint state dict
        gnn_ckpt = ckpt_mgr.load("gnn_trained")
        gnn_cfg = config.gnn
        gnn_sd = gnn_ckpt["model_state_dict"]
        if "node_proj.0.weight" in gnn_sd:
            gnn_cfg.node_feature_dim = gnn_sd["node_proj.0.weight"].shape[1]
            gnn_cfg.hidden_dim = gnn_sd["node_proj.0.weight"].shape[0]
        if "conv_layers.0.att" in gnn_sd:
            gnn_cfg.num_heads = gnn_sd["conv_layers.0.att"].shape[1]
        # Count conv layers
        layer_idx = 0
        while f"conv_layers.{layer_idx}.bias" in gnn_sd:
            layer_idx += 1
        if layer_idx > 0:
            gnn_cfg.num_layers = layer_idx
        if "resistance_head.3.weight" in gnn_sd:
            gnn_cfg.num_drugs = gnn_sd["resistance_head.3.weight"].shape[0]
        gnn = ResistanceGNN(gnn_cfg).to(device)
        gnn.load_state_dict(gnn_sd)
        logger.info("Loaded GNN from gnn_trained checkpoint (hidden=%d, layers=%d, drugs=%d)",
                     gnn_cfg.hidden_dim, gnn_cfg.num_layers, gnn_cfg.num_drugs)

        drug_names = config.data.target_drugs[:gnn_cfg.num_drugs]

        return cls(vae, scorer, gnn, config, drug_names)

    @torch.no_grad()
    def predict_single(
        self,
        proteomics: torch.Tensor,
        protein_names: list[str],
    ) -> PredictionResult:
        """Run inference on a single proteomic profile.

        Args:
            proteomics: (P,) protein abundance vector.
            protein_names: List of P protein names matching the vector.

        Returns:
            PredictionResult with all predictions.
        """
        # Ensure batch dimension
        x = proteomics.unsqueeze(0).to(self.device)

        # Module 1: Infer memory state
        recon, mu, log_var = self.vae(x)
        memory_state = mu.squeeze(0)  # (latent_dim,)

        # Module 2: Compute stability score
        stability = self.stability_scorer(x, protein_names)
        stability_score = stability.item()

        # Module 3: Predict drug resistance
        # Build minimal graph for single sample
        # In production, use full PPI graph
        from myelomemory.data.graph_builder import build_sample_graph

        num_proteins = x.shape[1]
        edge_index = torch.stack([
            torch.arange(num_proteins - 1),
            torch.arange(1, num_proteins),
        ]).to(self.device)
        edge_attr = torch.ones(num_proteins - 1, 1).to(self.device)

        graph = build_sample_graph(
            proteomics=x.squeeze(0),
            memory_state=memory_state,
            stability_score=stability_score,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        graph = graph.to(self.device)

        # Wrap in batch
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([graph])

        gnn_output = self.gnn(batch)

        resistance_values = gnn_output["resistance"].squeeze(0).cpu()
        drug_resistance = {
            name: resistance_values[i].item()
            for i, name in enumerate(self.drug_names)
        }

        drug_reversibility = {}
        if gnn_output.get("reversibility") is not None:
            rev_logits = gnn_output["reversibility"].squeeze(0).cpu()
            drug_reversibility = {
                name: torch.sigmoid(rev_logits[i]).item()
                for i, name in enumerate(self.drug_names)
            }

        return PredictionResult(
            memory_state=memory_state.cpu(),
            stability_score=stability_score,
            drug_resistance=drug_resistance,
            drug_reversibility=drug_reversibility,
            reconstructed_epigenome=recon.squeeze(0).cpu(),
        )

    @torch.no_grad()
    def predict(
        self,
        dataset: Any,
        split: str = "test",
    ) -> list[PredictionResult]:
        """Run inference on a dataset split.

        Args:
            dataset: MultiOmicsDataset.
            split: Which split to predict on.

        Returns:
            List of PredictionResult, one per sample in the split.
        """
        results = []
        protein_names = dataset.protein_names

        for i in range(len(dataset)):
            proteomics = dataset.proteomics[i]
            result = self.predict_single(proteomics, protein_names)
            results.append(result)

        return results
