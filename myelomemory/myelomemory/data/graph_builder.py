"""Build PyTorch Geometric graph objects from STRING PPI network.

Constructs per-sample graphs where nodes are proteins and edges are
protein-protein interactions. Node features are composed from proteomics,
VAE latent embeddings, and stability scores.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency during data prep stages
try:
    from torch_geometric.data import Data, Batch
except ImportError:
    Data = None
    Batch = None


def build_protein_index(protein_names: list[str]) -> dict[str, int]:
    """Create a mapping from protein name to integer index.

    Args:
        protein_names: List of protein/gene names from the proteomics data.

    Returns:
        Dict mapping protein name → integer index.
    """
    return {name: idx for idx, name in enumerate(protein_names)}


def build_edge_index(
    ppi_edges: list[tuple[str, str]],
    ppi_scores: list[float],
    protein_index: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert STRING PPI edges to PyG edge_index and edge_attr tensors.

    Only includes edges where BOTH proteins are in the protein index
    (i.e., present in the proteomics data).

    Args:
        ppi_edges: List of (protein1, protein2) tuples.
        ppi_scores: List of confidence scores for each edge.
        protein_index: Mapping from protein name → index.

    Returns:
        Tuple of (edge_index [2, E], edge_attr [E, 1]).
    """
    src, dst, weights = [], [], []

    for (p1, p2), score in zip(ppi_edges, ppi_scores):
        if p1 in protein_index and p2 in protein_index:
            i, j = protein_index[p1], protein_index[p2]
            # Undirected: add both directions
            src.extend([i, j])
            dst.extend([j, i])
            weights.extend([score, score])

    if not src:
        logger.warning("No PPI edges matched protein names. Check naming conventions.")
        # Return minimal self-loop graph
        n = len(protein_index)
        edge_index = torch.stack([torch.arange(n), torch.arange(n)])
        edge_attr = torch.ones(n, 1)
        return edge_index, edge_attr

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    logger.info(
        f"PPI graph: {len(protein_index)} nodes, {edge_index.shape[1]} directed edges"
    )

    return edge_index, edge_attr


def build_sample_graph(
    proteomics: torch.Tensor,
    memory_state: torch.Tensor | None,
    stability_score: float | None,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    drug_sensitivity: torch.Tensor | None = None,
) -> "Data":
    """Build a single PyG Data object for one sample.

    Node features are constructed by concatenating:
        - Protein abundance from proteomics (P dims)
        - VAE latent memory state (L dims, broadcast to all nodes)
        - Stability score (1 dim, broadcast to all nodes)

    Args:
        proteomics: (P,) protein abundance vector for this sample.
        memory_state: (L,) VAE latent vector, or None if not yet computed.
        stability_score: Scalar stability score, or None.
        edge_index: (2, E) PPI edge indices (shared across samples).
        edge_attr: (E, 1) edge weights.
        drug_sensitivity: (D,) IC50 values for target drugs.

    Returns:
        PyG Data object with node features, edges, and labels.
    """
    if Data is None:
        raise ImportError("torch_geometric is required. Install with: pip install torch-geometric")

    num_nodes = proteomics.shape[0]
    features = [proteomics.unsqueeze(1)]  # (P, 1) — one feature per protein node

    # For the GNN, each node gets the sample's proteomics value as its primary feature.
    # We also broadcast the global memory state and stability score to all nodes.
    node_x = proteomics.unsqueeze(1)  # (P, 1)

    if memory_state is not None:
        # Broadcast latent vector: (L,) → (P, L)
        memory_broadcast = memory_state.unsqueeze(0).expand(num_nodes, -1)
        node_x = torch.cat([node_x, memory_broadcast], dim=1)

    if stability_score is not None:
        # Broadcast scalar: () → (P, 1)
        stab_broadcast = torch.full((num_nodes, 1), stability_score, device=node_x.device)
        node_x = torch.cat([node_x, stab_broadcast], dim=1)

    data = Data(
        x=node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    if drug_sensitivity is not None:
        data.y = drug_sensitivity

    return data


def build_batch_graphs(
    proteomics_batch: torch.Tensor,
    memory_states: torch.Tensor | None,
    stability_scores: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    drug_sensitivity_batch: torch.Tensor | None = None,
) -> "Batch":
    """Build a batched PyG graph from multiple samples.

    Args:
        proteomics_batch: (B, P) batch of protein abundance vectors.
        memory_states: (B, L) batch of VAE latent vectors, or None.
        stability_scores: (B,) batch of stability scores, or None.
        edge_index: (2, E) shared PPI edge indices.
        edge_attr: (E, 1) shared edge weights.
        drug_sensitivity_batch: (B, D) batch of IC50 targets, or None.

    Returns:
        PyG Batch object.
    """
    if Batch is None:
        raise ImportError("torch_geometric is required.")

    graphs = []
    batch_size = proteomics_batch.shape[0]

    for i in range(batch_size):
        mem = memory_states[i] if memory_states is not None else None
        stab = stability_scores[i].item() if stability_scores is not None else None
        drug = drug_sensitivity_batch[i] if drug_sensitivity_batch is not None else None

        g = build_sample_graph(
            proteomics=proteomics_batch[i],
            memory_state=mem,
            stability_score=stab,
            edge_index=edge_index,
            edge_attr=edge_attr,
            drug_sensitivity=drug,
        )
        graphs.append(g)

    return Batch.from_data_list(graphs)
