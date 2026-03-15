"""Integration tests for the full MyeloMemory pipeline.

Tests load real checkpoints and verify end-to-end inference produces
correct shapes, ranges, and deterministic outputs.

Requires: checkpoints/ directory with trained model files.
Skip gracefully if checkpoints are not available.
"""

import pytest
import torch

# Skip all tests if checkpoints don't exist
pytestmark = pytest.mark.skipif(
    not all(
        __import__("pathlib").Path(f"checkpoints/{name}").exists()
        for name in ["vae_finetuned.pt", "stability_calibrated.pt", "gnn_trained.pt", "data_ready.pt"]
    ),
    reason="Trained checkpoints not available (run full pipeline first)",
)


@pytest.fixture(scope="module")
def pipeline():
    from myelomemory.config import MyeloMemoryConfig
    from myelomemory.inference.pipeline import MyeloMemoryPipeline
    from myelomemory.utils.checkpoint import CheckpointManager

    config = MyeloMemoryConfig()
    ckpt_mgr = CheckpointManager(config.checkpoint_dir)
    return MyeloMemoryPipeline.from_checkpoints(ckpt_mgr, config)


@pytest.fixture(scope="module")
def dataset():
    ckpt = torch.load("checkpoints/data_ready.pt", map_location="cpu", weights_only=False)
    return ckpt["dataset"]


class TestPipelineOutputs:
    """Test that pipeline outputs have correct shapes and ranges."""

    def test_predict_single_output_shape(self, pipeline, dataset):
        """Single prediction should return all expected fields."""
        result = pipeline.predict_single(dataset.proteomics[0], dataset.protein_names)

        assert result.memory_state.shape == (64,), "Memory state should be 64-dim"
        assert isinstance(result.stability_score, float)
        assert isinstance(result.drug_resistance, dict)
        assert isinstance(result.drug_reversibility, dict)

    def test_stability_score_range(self, pipeline, dataset):
        """Stability scores must be in [0, 1]."""
        result = pipeline.predict_single(dataset.proteomics[0], dataset.protein_names)
        assert 0.0 <= result.stability_score <= 1.0

    def test_drug_predictions_count(self, pipeline, dataset):
        """Should have predictions for each trained drug."""
        result = pipeline.predict_single(dataset.proteomics[0], dataset.protein_names)
        assert len(result.drug_resistance) == len(pipeline.drug_names)
        for drug in pipeline.drug_names:
            assert drug in result.drug_resistance

    def test_memory_state_64dim(self, pipeline, dataset):
        """Memory state embedding should be 64-dimensional."""
        result = pipeline.predict_single(dataset.proteomics[0], dataset.protein_names)
        assert result.memory_state.shape == (64,)
        assert torch.isfinite(result.memory_state).all()


class TestDeterminism:
    """Test that eval mode produces deterministic outputs."""

    def test_same_input_same_output(self, pipeline, dataset):
        """Same input should produce identical output in eval mode."""
        x = dataset.proteomics[0]
        names = dataset.protein_names

        r1 = pipeline.predict_single(x, names)
        r2 = pipeline.predict_single(x, names)

        assert abs(r1.stability_score - r2.stability_score) < 1e-5
        assert torch.allclose(r1.memory_state, r2.memory_state, atol=1e-5)

    def test_different_inputs_different_outputs(self, pipeline, dataset):
        """Different inputs should produce different outputs."""
        r1 = pipeline.predict_single(dataset.proteomics[0], dataset.protein_names)
        r2 = pipeline.predict_single(dataset.proteomics[10], dataset.protein_names)

        # At least memory states should differ
        assert not torch.allclose(r1.memory_state, r2.memory_state, atol=1e-3)


class TestMultipleSamples:
    """Test predictions on multiple samples."""

    def test_predict_10_samples(self, pipeline, dataset):
        """Should be able to predict on 10 samples without errors."""
        for i in range(10):
            result = pipeline.predict_single(dataset.proteomics[i], dataset.protein_names)
            assert 0.0 <= result.stability_score <= 1.0
            assert result.memory_state.shape == (64,)
            assert len(result.drug_resistance) > 0
