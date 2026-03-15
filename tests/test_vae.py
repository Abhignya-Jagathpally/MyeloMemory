"""Tests for Module 1: Proteome-to-Epigenome VAE.

Uses small tensors matching real data formats to verify shape contracts
and training logic. No external downloads needed for unit tests.
The pipeline itself requires real data from DepMap, ENCODE, etc.
"""

import pytest
import torch

from myelomemory.config import VAEConfig
from myelomemory.models.vae import ProteomeToEpigenomeVAE, _kl_divergence, _cyclical_kl_weight


@pytest.fixture
def small_config() -> VAEConfig:
    """Minimal VAE config for testing."""
    return VAEConfig(
        input_dim=100,
        epigenome_dim=200,
        latent_dim=16,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[32, 64],
        dropout=0.0,
        use_batch_norm=False,
        gradient_checkpointing=False,
        pretrain_epochs=2,
        finetune_epochs=1,
        batch_size=8,
        patience=5,
    )


@pytest.fixture
def model(small_config: VAEConfig) -> ProteomeToEpigenomeVAE:
    return ProteomeToEpigenomeVAE(small_config)


class TestVAEArchitecture:
    def test_forward_shape(self, model: ProteomeToEpigenomeVAE, small_config: VAEConfig) -> None:
        x = torch.randn(8, small_config.input_dim)
        recon, mu, log_var = model(x)

        assert recon.shape == (8, small_config.epigenome_dim)
        assert mu.shape == (8, small_config.latent_dim)
        assert log_var.shape == (8, small_config.latent_dim)

    def test_encode_shape(self, model: ProteomeToEpigenomeVAE, small_config: VAEConfig) -> None:
        x = torch.randn(4, small_config.input_dim)
        mu, log_var = model.encode(x)

        assert mu.shape == (4, small_config.latent_dim)
        assert log_var.shape == (4, small_config.latent_dim)

    def test_decode_shape(self, model: ProteomeToEpigenomeVAE, small_config: VAEConfig) -> None:
        z = torch.randn(4, small_config.latent_dim)
        recon = model.decode(z)

        assert recon.shape == (4, small_config.epigenome_dim)

    def test_get_memory_state_deterministic(self, model: ProteomeToEpigenomeVAE, small_config: VAEConfig) -> None:
        x = torch.randn(4, small_config.input_dim)
        state1 = model.get_memory_state(x)
        state2 = model.get_memory_state(x)

        assert torch.allclose(state1, state2), "Memory state should be deterministic"

    def test_reparameterize_stochastic_in_train(self, model: ProteomeToEpigenomeVAE) -> None:
        model.train()
        mu = torch.zeros(4, 16)
        log_var = torch.zeros(4, 16)

        samples = [model.reparameterize(mu, log_var) for _ in range(10)]
        # Not all samples should be identical (stochastic)
        different = any(not torch.allclose(samples[0], s) for s in samples[1:])
        assert different, "Reparameterize should be stochastic in train mode"

    def test_reparameterize_deterministic_in_eval(self, model: ProteomeToEpigenomeVAE) -> None:
        model.eval()
        mu = torch.randn(4, 16)
        log_var = torch.randn(4, 16)

        z = model.reparameterize(mu, log_var)
        assert torch.allclose(z, mu), "Reparameterize should return mu in eval mode"


class TestKLDivergence:
    def test_zero_kl_for_standard_normal(self) -> None:
        mu = torch.zeros(10, 16)
        log_var = torch.zeros(10, 16)
        kl = _kl_divergence(mu, log_var)
        assert abs(kl) < 1e-5

    def test_positive_kl_for_non_standard(self) -> None:
        mu = torch.ones(10, 16)
        log_var = torch.ones(10, 16)
        kl = _kl_divergence(mu, log_var)
        assert kl > 0


class TestCyclicalKLWeight:
    def test_starts_at_zero(self) -> None:
        weight = _cyclical_kl_weight(0, 1000, 4, 0.5, 1.0)
        assert weight == 0.0

    def test_reaches_max(self) -> None:
        weight = _cyclical_kl_weight(124, 1000, 4, 0.5, 1.0)
        assert weight == pytest.approx(1.0, abs=0.01)

    def test_cycles(self) -> None:
        # End of first cycle should be at max weight
        weight_end_cycle = _cyclical_kl_weight(249, 1000, 4, 0.5, 1.0)
        # Start of new cycle (step 250) should reset to near zero
        weight_start_next = _cyclical_kl_weight(250, 1000, 4, 0.5, 1.0)
        assert weight_start_next < weight_end_cycle