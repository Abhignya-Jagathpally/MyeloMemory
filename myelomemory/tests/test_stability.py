"""Tests for Module 2: Memory Stability Scorer."""

import pytest
import torch

from myelomemory.config import StabilityConfig
from myelomemory.models.stability import MemoryStabilityScorer, ChromatinODE


@pytest.fixture
def small_config() -> StabilityConfig:
    return StabilityConfig(
        reader_writer_proteins=["EZH2", "DNMT1", "TET2", "KDM6A"],
        n_perturbation_samples=5,
        integration_time=10.0,
        calibration_epochs=2,
        calibration_batch_size=4,
    )


@pytest.fixture
def scorer(small_config: StabilityConfig) -> MemoryStabilityScorer:
    return MemoryStabilityScorer(small_config)


class TestChromatinoODE:
    def test_forward_shape(self, small_config: StabilityConfig) -> None:
        ode = ChromatinODE(small_config)
        t = torch.tensor(0.0)
        # State: [a, r, 8 ODE params]
        state = torch.randn(4, 10).abs()  # All positive
        dstate = ode(t, state)
        assert dstate.shape == (4, 10)

    def test_param_derivatives_zero(self, small_config: StabilityConfig) -> None:
        ode = ChromatinODE(small_config)
        t = torch.tensor(0.0)
        state = torch.randn(4, 10).abs()
        dstate = ode(t, state)
        # Last 8 dims (params) should have zero derivatives
        assert torch.allclose(dstate[:, 2:], torch.zeros(4, 8))


class TestStabilityScorer:
    def test_extract_reader_writer_levels(self, scorer: MemoryStabilityScorer) -> None:
        all_names = ["GeneA", "EZH2", "GeneB", "DNMT1", "TET2", "KDM6A", "GeneC"]
        proteomics = torch.randn(2, len(all_names))

        rw = scorer.extract_reader_writer_levels(proteomics, all_names)
        assert rw.shape == (2, 4)
        # EZH2 is at index 1 in all_names
        assert torch.allclose(rw[:, 0], proteomics[:, 1])

    def test_missing_proteins_get_zero(self, scorer: MemoryStabilityScorer) -> None:
        all_names = ["GeneA", "GeneB"]  # None of the reader/writers
        proteomics = torch.randn(2, 2)

        rw = scorer.extract_reader_writer_levels(proteomics, all_names)
        assert rw.shape == (2, 4)
        assert torch.allclose(rw, torch.zeros(2, 4))

    def test_output_in_range(self, scorer: MemoryStabilityScorer) -> None:
        pytest.importorskip("torchdiffeq")

        all_names = ["EZH2", "DNMT1", "TET2", "KDM6A"] + [f"Gene{i}" for i in range(96)]
        proteomics = torch.randn(2, 100).abs()

        scores = scorer(proteomics, all_names)
        assert scores.shape == (2,)
        assert (scores >= 0).all() and (scores <= 1).all()
