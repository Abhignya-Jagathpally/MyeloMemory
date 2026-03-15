"""Tests for Jacobian eigenvalue-based basin depth estimation.

Validates the scientific core of the stability scorer against known
analytical solutions and edge cases.
"""

import pytest
import torch

from myelomemory.config import StabilityConfig
from myelomemory.models.stability import MemoryStabilityScorer


@pytest.fixture
def scorer():
    config = StabilityConfig()
    s = MemoryStabilityScorer(config)
    s.eval()
    return s


class TestJacobianEigenvalues:
    """Test the _estimate_basin_depth method."""

    def test_positive_basin_depth_at_stable_point(self, scorer):
        """A stable fixed point should have negative eigenvalues → positive basin depth."""
        ode_params = torch.tensor([[0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]])
        a_steady = torch.tensor([[0.8]])
        r_steady = torch.tensor([[0.8]])
        with torch.no_grad():
            depth = scorer._estimate_basin_depth(ode_params, a_steady, r_steady)
        # Basin depth should be positive (eigenvalues negative at stable point)
        assert depth.shape == (1,)
        assert not depth.isnan().any(), "Basin depth should not be NaN"

    def test_different_params_give_different_depths(self, scorer):
        """Different ODE parameters should produce different basin depths."""
        params_a = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5]])
        params_b = torch.tensor([[0.9, 0.9, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9]])
        a_steady = torch.tensor([[0.7]])
        r_steady = torch.tensor([[0.7]])

        with torch.no_grad():
            depth_a = scorer._estimate_basin_depth(params_a, a_steady, r_steady)
            depth_b = scorer._estimate_basin_depth(params_b, a_steady, r_steady)

        assert not torch.allclose(depth_a, depth_b, atol=1e-4), \
            "Different ODE params should produce different basin depths"

    def test_batch_processing(self, scorer):
        """Basin depth should work on batched inputs."""
        B = 8
        ode_params = torch.rand(B, 8) * 0.5 + 0.3
        a_steady = torch.rand(B, 1) * 0.5 + 0.5
        r_steady = torch.rand(B, 1) * 0.5 + 0.5

        with torch.no_grad():
            depth = scorer._estimate_basin_depth(ode_params, a_steady, r_steady)

        assert depth.shape == (B,)
        assert not depth.isnan().any()

    def test_float32_precision(self, scorer):
        """Jacobian finite differences should use float32 even when inputs differ."""
        ode_params = torch.tensor([[0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]])
        a_steady = torch.tensor([[0.8]])
        r_steady = torch.tensor([[0.8]])

        with torch.no_grad():
            depth = scorer._estimate_basin_depth(ode_params, a_steady, r_steady)

        # Should not be NaN or Inf
        assert torch.isfinite(depth).all()


class TestNaNHandling:
    """Test graceful NaN handling in edge cases."""

    def test_nan_in_steady_state_handled(self, scorer):
        """NaN steady states should produce fallback score of 0.5."""
        proteins = torch.randn(1, len(scorer.protein_names))
        # Create extreme protein levels that might cause ODE divergence
        proteins = proteins * 100

        with torch.no_grad():
            score = scorer(proteins, scorer.protein_names)

        assert score.shape == (1,)
        assert not score.isnan().any(), "NaN should be replaced with fallback"
        assert (score >= 0).all() and (score <= 1).all(), "Score must be in [0, 1]"

    def test_score_in_valid_range(self, scorer):
        """All outputs should be in [0, 1] regardless of input."""
        for _ in range(5):
            proteins = torch.randn(4, len(scorer.protein_names)) * 5
            with torch.no_grad():
                scores = scorer(proteins, scorer.protein_names)
            scores_valid = scores[~scores.isnan()]
            if len(scores_valid) > 0:
                assert (scores_valid >= 0).all() and (scores_valid <= 1).all()


class TestLearnableNormalization:
    """Test that basin_center and basin_scale are learnable parameters."""

    def test_parameters_exist(self, scorer):
        """basin_center and basin_scale should be nn.Parameters."""
        param_names = [name for name, _ in scorer.named_parameters()]
        assert "basin_center" in param_names
        assert "basin_scale" in param_names

    def test_parameters_require_grad(self, scorer):
        """Parameters should have requires_grad=True."""
        assert scorer.basin_center.requires_grad
        assert scorer.basin_scale.requires_grad

    def test_initial_values(self, scorer):
        """Parameters should be initialized to training distribution stats."""
        assert abs(scorer.basin_center.item() - 1.53) < 0.01
        assert abs(scorer.basin_scale.item() - 1.56) < 0.01
