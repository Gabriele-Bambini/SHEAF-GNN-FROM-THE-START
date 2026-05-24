"""Unit tests for FiLM (Feature-wise Linear Modulation) allele conditioning."""

import pytest
import torch

from tsgnn.model.film import AlleleFiLM, ZeroFiLM


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def film():
    return AlleleFiLM(esm_dim=1280, conditioning_dim=128, modulation_dim=32)


@pytest.fixture
def zero_film():
    return ZeroFiLM()


# ── Shape Tests ───────────────────────────────────────────────────────────

def test_output_shape_1d_embedding(film):
    """Output shape should match GRU output shape with 1D embedding."""
    E, mod_dim = 50, 32
    esm_emb = torch.randn(1280)
    gru_out = torch.randn(E, mod_dim)
    result = film(esm_emb, gru_out)
    assert result.shape == (E, mod_dim)


def test_output_shape_2d_embedding(film):
    """Output shape should match GRU output shape with 2D embedding."""
    E, mod_dim = 50, 32
    esm_emb = torch.randn(1, 1280)
    gru_out = torch.randn(E, mod_dim)
    result = film(esm_emb, gru_out)
    assert result.shape == (E, mod_dim)


# ── Identity Initialization Tests ─────────────────────────────────────────

def test_identity_init_with_zero_embedding(film):
    """With zero ESM embedding, FiLM should act as near-identity at initialization.

    With small Gaussian weights (std=0.01) on the last film_generator layer,
    the modulation is small but not exactly zero. The allele_encoder biases
    can produce non-zero intermediate values even for a zero input, so we
    check only that the output is 'close' to the GRU output — not exact.

    The essential property: FiLM does not dramatically corrupt inputs at init.
    """
    E, mod_dim = 50, 32
    gru_out = torch.randn(E, mod_dim)
    zero_emb = torch.zeros(1280)
    result = film(zero_emb, gru_out)
    # gamma ≈ 1 + small, beta ≈ small → output ≈ gru_out with small perturbation
    # Relative change should be small compared to the signal magnitude
    relative_diff = (result - gru_out).norm() / (gru_out.norm() + 1e-8)
    assert relative_diff < 0.5, (
        f"FiLM at init changes input too much: relative diff = {relative_diff:.4f}"
    )



def test_zero_film_is_identity(zero_film):
    """ZeroFiLM should always return the input unchanged."""
    E, mod_dim = 50, 32
    esm_emb = torch.randn(1280)
    gru_out = torch.randn(E, mod_dim)
    result = zero_film(esm_emb, gru_out)
    assert torch.equal(result, gru_out)


# ── Gradient Flow Tests ──────────────────────────────────────────────────

def test_gradient_flows_to_esm_embedding(film):
    """Gradient must flow back through FiLM to the ESM embedding.

    Theoretical justification: FiLM modulates via gamma * x + beta, where
    gamma and beta are functions of z_allele. Therefore dL/dz ≠ 0 whenever
    the GRU output is nonzero, ensuring the allele embedding is trainable.
    """
    E, mod_dim = 50, 32
    esm_emb = torch.randn(1280, requires_grad=True)
    gru_out = torch.randn(E, mod_dim)
    result = film(esm_emb, gru_out)
    result.sum().backward()
    assert esm_emb.grad is not None
    assert esm_emb.grad.norm() > 0


def test_gradient_flows_to_gru_output(film):
    """Gradient must flow back through FiLM to the GRU output."""
    E, mod_dim = 50, 32
    esm_emb = torch.randn(1280)
    gru_out = torch.randn(E, mod_dim, requires_grad=True)
    result = film(esm_emb, gru_out)
    result.sum().backward()
    assert gru_out.grad is not None
    assert gru_out.grad.norm() > 0


# ── Behavioral Tests ─────────────────────────────────────────────────────

def test_different_alleles_produce_different_outputs(film):
    """Different ESM embeddings should produce different modulations.

    This validates that FiLM actually conditions on the allele identity
    rather than ignoring it. Two random embeddings should produce
    statistically different gamma/beta parameters.
    """
    E, mod_dim = 50, 32
    gru_out = torch.randn(E, mod_dim)
    emb_a = torch.randn(1280)
    emb_b = torch.randn(1280)
    result_a = film(emb_a, gru_out)
    result_b = film(emb_b, gru_out)
    assert not torch.allclose(result_a, result_b, atol=1e-3)


def test_conditioning_vector_shape(film):
    """The intermediate conditioning vector should have expected shape."""
    esm_emb = torch.randn(1280)
    z = film.get_conditioning_vector(esm_emb)
    assert z.shape == (1, 128)  # conditioning_dim
