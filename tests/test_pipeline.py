"""Integration test for the full MyeloMemory pipeline.

Verifies that all modules connect correctly using small tensors in the
same format as real data. These are UNIT TESTS for shape contracts and
module wiring — not substitutes for real data. The pipeline requires
actual datasets from DepMap, ENCODE, STRING, and GDSC.
"""

import pytest
import torch

from myelomemory.config import MyeloMemoryConfig, VAEConfig, StabilityConfig, GNNConfig
from myelomemory.models.vae import ProteomeToEpigenomeVAE
from myelomemory.utils.checkpoint import CheckpointManager


@pytest.fixture
def tiny_config() -> MyeloMemoryConfig:
    """Minimal config for integration testing."""
    config = MyeloMemoryConfig()
    config.vae = VAEConfig(
        input_dim=50,
        epigenome_dim=100,
        latent_dim=8,
        encoder_hidden_dims=[32],
        decoder_hidden_dims=[32],
        dropout=0.0,
        use_batch_norm=False,
        gradient_checkpointing=False,
    )
    config.stability = StabilityConfig(
        reader_writer_proteins=["EZH2", "DNMT1"],
        n_perturbation_samples=3,
        integration_time=5.0,
    )
    config.gnn = GNNConfig(
        node_feature_dim=50 + 8 + 1,  # proteomics + latent + stability
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        num_drugs=2,
    )
    return config


class TestCheckpointManager:
    def test_save_and_load(self, tmp_path: pytest.TempPathFactory) -> None:
        import logging
        mgr = CheckpointManager(tmp_path, logging.getLogger())

        data = {"model_state_dict": {"weight": torch.randn(3, 3)}, "epoch": 5}
        mgr.save("test_stage", data)

        assert mgr.exists("test_stage")

        loaded = mgr.load("test_stage")
        assert loaded["epoch"] == 5
        assert torch.allclose(loaded["model_state_dict"]["weight"], data["model_state_dict"]["weight"])

    def test_nonexistent_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        import logging
        mgr = CheckpointManager(tmp_path, logging.getLogger())

        with pytest.raises(FileNotFoundError):
            mgr.load("nonexistent")


class TestVAEToStabilityConnection:
    def test_vae_output_feeds_stability(self, tiny_config: MyeloMemoryConfig) -> None:
        pytest.importorskip("torchdiffeq")
        from myelomemory.models.stability import MemoryStabilityScorer

        vae = ProteomeToEpigenomeVAE(tiny_config.vae)
        scorer = MemoryStabilityScorer(tiny_config.stability)

        # Small test tensor matching real data format
        protein_names = ["EZH2", "DNMT1"] + [f"Gene{i}" for i in range(48)]
        proteomics = torch.randn(2, 50).abs()

        # VAE extracts memory state
        memory_state = vae.get_memory_state(proteomics)
        assert memory_state.shape == (2, 8)

        # Stability scorer processes the same proteomics
        scores = scorer(proteomics, protein_names)
        assert scores.shape == (2,)
        assert (scores >= 0).all() and (scores <= 1).all()