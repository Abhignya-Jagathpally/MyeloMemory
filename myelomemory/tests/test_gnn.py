"""Tests for Module 3: Resistance Pathway GNN."""

import pytest
import torch

from myelomemory.config import GNNConfig


@pytest.fixture
def small_config() -> GNNConfig:
    return GNNConfig(
        node_feature_dim=10,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_drugs=3,
        predict_reversibility=True,
        epochs=2,
        batch_size=4,
        patience=5,
    )


class TestResistanceGNN:
    @pytest.mark.skipif(
        not pytest.importorskip("torch_geometric", reason="PyG not installed"),
        reason="torch_geometric required",
    )
    def test_forward_shape(self, small_config: GNNConfig) -> None:
        from myelomemory.models.gnn import ResistanceGNN
        from torch_geometric.data import Data, Batch

        model = ResistanceGNN(small_config)

        # Build small test graphs
        graphs = []
        for _ in range(4):
            num_nodes = 20
            edge_index = torch.randint(0, num_nodes, (2, 40))
            x = torch.randn(num_nodes, small_config.node_feature_dim)
            edge_attr = torch.rand(40, 1)
            g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graphs.append(g)

        batch = Batch.from_data_list(graphs)
        output = model(batch)

        assert output["resistance"].shape == (4, small_config.num_drugs)
        assert output["reversibility"].shape == (4, small_config.num_drugs)

    @pytest.mark.skipif(
        not pytest.importorskip("torch_geometric", reason="PyG not installed"),
        reason="torch_geometric required",
    )
    def test_no_reversibility_head(self, small_config: GNNConfig) -> None:
        from myelomemory.models.gnn import ResistanceGNN
        from torch_geometric.data import Data, Batch

        small_config.predict_reversibility = False
        model = ResistanceGNN(small_config)

        graphs = []
        for _ in range(2):
            num_nodes = 10
            edge_index = torch.randint(0, num_nodes, (2, 20))
            x = torch.randn(num_nodes, small_config.node_feature_dim)
            edge_attr = torch.rand(20, 1)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        batch = Batch.from_data_list(graphs)
        output = model(batch)

        assert "resistance" in output
        assert output.get("reversibility") is None
