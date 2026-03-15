"""Module 3: Resistance Pathway GNN — drug resistance prediction on PPI networks.

A Graph Attention Network (GAT) that operates on the STRING protein-protein
interaction network, with node features composed from:
    - Protein abundance (from proteomics)
    - Inferred memory state (from VAE latent, broadcast to all nodes)
    - Stability score (from ODE model, broadcast to all nodes)

Predicts:
    - Per-drug IC50 (resistance probability for each target drug)
    - Binary reversibility flag (is this resistance transient or locked-in?)

The key innovation: existing drug sensitivity predictors use static features.
MyeloMemory adds DYNAMIC features (memory state + stability) that capture
whether resistance is a transient adaptation or permanent epigenetic memory.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from myelomemory.config import GNNConfig
from myelomemory.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

# Lazy imports for PyG
try:
    from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.nn import GlobalAttention
    from torch_geometric.data import Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class ResistanceGNN(nn.Module):
    """Graph neural network for drug resistance prediction.

    Architecture:
        1. Node feature projection
        2. N layers of graph convolution (GAT/GCN/GraphSAGE)
        3. Global graph pooling (attention-based or mean)
        4. Dual prediction heads:
           - Resistance head: graph embedding → per-drug IC50
           - Reversibility head: graph embedding → per-drug binary (reversible/locked)

    Args:
        config: GNNConfig with architecture and training hyperparameters.
    """

    def __init__(self, config: GNNConfig) -> None:
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required. Install with: "
                "pip install torch-geometric"
            )

        super().__init__()
        self.config = config

        # Node feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()

        for i in range(config.num_layers):
            in_dim = config.hidden_dim
            out_dim = config.hidden_dim

            if config.conv_type == "gat":
                # GAT with multi-head attention
                assert out_dim % config.num_heads == 0
                conv = GATv2Conv(
                    in_dim,
                    out_dim // config.num_heads,
                    heads=config.num_heads,
                    dropout=config.dropout,
                    concat=True,
                    edge_dim=1,  # Edge weight from STRING confidence
                )
            elif config.conv_type == "gcn":
                conv = GCNConv(in_dim, out_dim)
            elif config.conv_type == "graphsage":
                conv = SAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown conv type: {config.conv_type}")

            self.conv_layers.append(conv)
            self.conv_norms.append(nn.LayerNorm(out_dim))

        # Global pooling
        if config.pool_type == "global_attention":
            gate_nn = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            self.pool = GlobalAttention(gate_nn=gate_nn)
        else:
            self.pool = global_mean_pool

        # Resistance prediction head: graph → per-drug IC50
        self.resistance_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_drugs),
        )

        # Reversibility prediction head: graph → per-drug binary
        if config.predict_reversibility:
            self.reversibility_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_drugs),
            )
        else:
            self.reversibility_head = None

    def forward(self, batch: "Batch") -> dict[str, torch.Tensor]:
        """Forward pass on a batched graph.

        Args:
            batch: PyG Batch object with:
                - x: (total_nodes, node_feature_dim) node features
                - edge_index: (2, total_edges) edge indices
                - edge_attr: (total_edges, 1) edge weights
                - batch: (total_nodes,) batch assignment vector

        Returns:
            Dict with keys:
                - 'resistance': (B, num_drugs) predicted IC50 values
                - 'reversibility': (B, num_drugs) reversibility logits (if enabled)
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_vec = batch.batch

        # Project node features
        x = self.node_proj(x)

        # Graph convolution layers with residual connections
        for conv, norm in zip(self.conv_layers, self.conv_norms):
            if self.config.conv_type == "gat":
                x_new = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x_new = conv(x, edge_index)

            x_new = F.gelu(x_new)
            x_new = F.dropout(x_new, p=self.config.dropout, training=self.training)

            # Residual connection
            x = norm(x + x_new)

        # Global pooling: (total_nodes, H) → (B, H)
        if self.config.pool_type == "global_attention":
            graph_emb = self.pool(x, batch_vec)
        else:
            graph_emb = self.pool(x, batch_vec)

        # Prediction heads
        result = {
            "resistance": self.resistance_head(graph_emb),
        }

        if self.reversibility_head is not None:
            result["reversibility"] = self.reversibility_head(graph_emb)

        return result


def _resistance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE loss for drug resistance prediction.

    Args:
        pred: (B, D) predicted IC50.
        target: (B, D) true IC50 (may contain NaN).
        mask: (B, D) boolean mask (True where target is valid).

    Returns:
        Scalar loss.
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(pred[mask], target[mask])


def _reversibility_loss(
    pred: torch.Tensor,
    stability_scores: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Binary cross-entropy loss for reversibility prediction.

    Uses stability score as a pseudo-label:
        stability < threshold → reversible (label=1)
        stability >= threshold → locked-in (label=0)

    Args:
        pred: (B, D) reversibility logits.
        stability_scores: (B,) stability scores from Module 2.
        threshold: Cutoff for reversible vs. locked-in.

    Returns:
        Scalar loss.
    """
    labels = (stability_scores < threshold).float()
    # Broadcast to all drugs
    labels = labels.unsqueeze(1).expand_as(pred)
    return F.binary_cross_entropy_with_logits(pred, labels)


def train_resistance_gnn(
    model: nn.Module,
    dataset: Any,
    splits: dict[str, list[int]],
    vae_checkpoint: dict[str, Any],
    stability_checkpoint: dict[str, Any],
    config: GNNConfig,
    ckpt_mgr: CheckpointManager,
    vae_config: Any = None,
    stability_config: Any = None,
) -> dict[str, Any]:
    """Train the resistance GNN with memory-augmented node features.

    Args:
        model: ResistanceGNN (possibly DDP-wrapped).
        dataset: MultiOmicsDataset.
        splits: Train/val/test splits.
        vae_checkpoint: Loaded VAE checkpoint.
        stability_checkpoint: Loaded stability scorer checkpoint.
        config: GNN hyperparameters.
        ckpt_mgr: Checkpoint manager.

    Returns:
        Dict with 'checkpoint_path' and 'metrics'.
    """
    from myelomemory.models.vae import ProteomeToEpigenomeVAE
    from myelomemory.models.stability import MemoryStabilityScorer
    from myelomemory.data.graph_builder import (
        build_protein_index, build_edge_index, build_sample_graph,
    )
    from torch_geometric.data import Batch as PyGBatch

    device = next(model.parameters()).device

    # --- Load frozen VAE for memory state extraction ---
    logger.info("Loading frozen VAE for memory state extraction")
    from myelomemory.config import VAEConfig
    if vae_config is not None:
        vae_cfg = vae_config
    else:
        vae_cfg_dict = vae_checkpoint.get("config", {})
        vae_cfg = VAEConfig(**vae_cfg_dict) if isinstance(vae_cfg_dict, dict) and vae_cfg_dict else VAEConfig()
    vae = ProteomeToEpigenomeVAE(vae_cfg).to(device)
    vae.load_state_dict(vae_checkpoint["model_state_dict"])
    vae.eval()

    # --- Load frozen stability scorer ---
    logger.info("Loading frozen stability scorer")
    from myelomemory.config import StabilityConfig
    if stability_config is not None:
        stab_cfg = stability_config
    else:
        stab_cfg_dict = stability_checkpoint.get("config", {})
        stab_cfg = StabilityConfig(**stab_cfg_dict) if isinstance(stab_cfg_dict, dict) and stab_cfg_dict else StabilityConfig()
    stability_scorer = MemoryStabilityScorer(stab_cfg).to(device)
    stability_scorer.load_state_dict(stability_checkpoint["model_state_dict"])
    stability_scorer.eval()

    # --- Build PPI graph (shared across all samples) ---
    logger.info("Building PPI graph from dataset")
    protein_index = build_protein_index(dataset.protein_names)
    if not hasattr(dataset, "ppi_edges") or dataset.ppi_edges is None:
        raise ValueError(
            "Dataset does not contain PPI edge data. Ensure the STRING PPI "
            "network was loaded during data preparation (scripts/download_data.sh)."
        )
    edge_index, edge_attr = build_edge_index(
        dataset.ppi_edges, dataset.ppi_scores, protein_index
    )
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    logger.info(f"PPI graph: {len(protein_index)} nodes, {edge_index.shape[1]} edges")

    # --- Pre-compute memory states and stability scores (frozen, no grad) ---
    logger.info("Pre-computing memory states and stability scores for all samples")
    all_memory_states = []
    all_stability_scores = []
    with torch.no_grad():
        for start in range(0, len(dataset), 128):
            end = min(start + 128, len(dataset))
            batch_prot = dataset.proteomics[start:end].to(device)

            memory = vae.get_memory_state(batch_prot)
            stability = stability_scorer(batch_prot, dataset.protein_names)

            all_memory_states.append(memory.cpu())
            all_stability_scores.append(stability.cpu())

    all_memory_states = torch.cat(all_memory_states, dim=0)   # (N, latent_dim)
    all_stability_scores = torch.cat(all_stability_scores, dim=0)  # (N,)
    logger.info(
        f"Pre-computed: {all_memory_states.shape[0]} memory states, "
        f"mean stability={all_stability_scores.mean():.3f}"
    )

    # --- Training loop ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )

    scaler = GradScaler("cuda")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        train_indices = splits["train"]
        for start in range(0, len(train_indices), config.batch_size):
            batch_idx = train_indices[start:start + config.batch_size]

            proteomics = dataset.proteomics[batch_idx].to(device)
            drug_targets = dataset.drug_sensitivity[batch_idx].to(device)
            drug_mask = ~torch.isnan(drug_targets)
            drug_targets_clean = torch.where(
                drug_mask, drug_targets, torch.zeros_like(drug_targets)
            )

            memory_states = all_memory_states[batch_idx].to(device)
            stability_scores = all_stability_scores[batch_idx].to(device)

            # Build per-sample graphs with real PPI structure and memory features
            graphs = []
            for i in range(len(batch_idx)):
                g = build_sample_graph(
                    proteomics=proteomics[i],
                    memory_state=memory_states[i],
                    stability_score=stability_scores[i].item(),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    drug_sensitivity=drug_targets_clean[i],
                )
                graphs.append(g)

            batch = PyGBatch.from_data_list(graphs)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)

                targets = torch.stack([g.y for g in graphs])
                loss_res = _resistance_loss(output["resistance"], targets, drug_mask)
                loss = config.resistance_loss_weight * loss_res

                if output.get("reversibility") is not None:
                    loss_rev = _reversibility_loss(
                        output["reversibility"], stability_scores
                    )
                    loss = loss + config.reversibility_loss_weight * loss_rev

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"[gnn_train] Epoch {epoch + 1}/{config.epochs} "
                f"loss={avg_loss:.4f}"
            )

        # Save best
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_counter = 0
            raw_model = model.module if hasattr(model, "module") else model
            ckpt_mgr.save("gnn_trained", {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
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
    }

    return {
        "checkpoint_path": ckpt_mgr.path("gnn_trained"),
        "metrics": metrics,
    }