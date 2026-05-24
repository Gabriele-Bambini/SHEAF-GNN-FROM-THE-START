"""Unit tests for the full TS-GNN architecture."""

import pytest
import torch

from tsgnn.model.tsgnn import TSGNN, create_tsgnn_from_config


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def model_params():
    """Canonical small-scale model parameters for testing."""
    return dict(N=20, E=60, d=4, input_dim=20, esm_dim=1280, K=5)


@pytest.fixture
def model(model_params):
    p = model_params
    edge_index = torch.randint(0, p["N"], (2, p["E"]))
    return TSGNN(
        num_nodes=p["N"],
        num_edges=p["E"],
        stalk_dim=p["d"],
        input_dim=p["input_dim"],
        esm_dim=p["esm_dim"],
        conditioning_dim=64,
        edge_index=edge_index,
        num_diffusion_steps=2,
    )


@pytest.fixture
def model_no_allele(model_params):
    """TS-GNN with FiLM disabled (ablation)."""
    p = model_params
    edge_index = torch.randint(0, p["N"], (2, p["E"]))
    return TSGNN(
        num_nodes=p["N"],
        num_edges=p["E"],
        stalk_dim=p["d"],
        input_dim=p["input_dim"],
        esm_dim=p["esm_dim"],
        conditioning_dim=64,
        edge_index=edge_index,
        num_diffusion_steps=2,
        use_allele_conditioning=False,
    )


# ── Forward Pass Tests ───────────────────────────────────────────────────

def test_forward_output_shapes(model, model_params):
    """Forward pass should return lists of correct shapes.

    Predictions: K tensors each (N, input_dim)
    Maps trajectory: K tensors each (E, 2, d, d)
    Laplacians: K tensors each (N*d, N*d)
    """
    p = model_params
    node_seq = torch.randn(p["K"], p["N"], p["input_dim"])
    allele_emb = torch.randn(p["esm_dim"])
    preds, maps, laps, stalks = model(node_seq, allele_emb, return_stalks=True)

    assert len(preds) == p["K"]
    assert len(maps) == p["K"]
    assert len(laps) == p["K"]
    assert len(stalks) == p["K"]

    for t in range(p["K"]):
        assert preds[t].shape == (p["N"], p["input_dim"])
        assert maps[t].shape == (p["E"], 2, p["d"], p["d"])
        assert laps[t].shape == (p["N"] * p["d"], p["N"] * p["d"])
        assert stalks[t].shape == (p["N"], p["d"])


def test_forward_no_nan(model, model_params):
    """Forward pass should produce finite values (no NaN/Inf).

    Theoretical concern: sheaf Laplacian multiplication can amplify values
    over multiple diffusion steps. Residual connections and proper init
    should prevent this.
    """
    p = model_params
    node_seq = torch.randn(p["K"], p["N"], p["input_dim"])
    allele_emb = torch.randn(p["esm_dim"])
    preds, _, laps = model(node_seq, allele_emb)

    for t in range(p["K"]):
        assert torch.isfinite(preds[t]).all(), f"NaN/Inf in prediction at t={t}"
        assert torch.isfinite(laps[t]).all(), f"NaN/Inf in Laplacian at t={t}"


def test_no_allele_forward(model_no_allele, model_params):
    """No-allele ablation model should still produce valid outputs."""
    p = model_params
    node_seq = torch.randn(p["K"], p["N"], p["input_dim"])
    allele_emb = torch.randn(p["esm_dim"])
    preds, maps, laps = model_no_allele(node_seq, allele_emb)
    assert len(preds) == p["K"]
    assert preds[0].shape == (p["N"], p["input_dim"])


# ── Gradient Flow Tests ──────────────────────────────────────────────────

def test_backward_no_error(model, model_params):
    """Full forward-backward pass should complete without errors."""
    p = model_params
    node_seq = torch.randn(p["K"], p["N"], p["input_dim"])
    allele_emb = torch.randn(p["esm_dim"])
    preds, _, _ = model(node_seq, allele_emb)
    loss = sum(p.sum() for p in preds)
    loss.backward()
    # At least some parameters should have gradients
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ── Decompose Rewiring Tests ─────────────────────────────────────────────

def test_decompose_rewiring_delta_nonzero(model, model_params):
    """The allele-specific delta Laplacians should be nonzero.

    Theoretical justification: with FiLM enabled, running the model with
    a nonzero allele embedding vs. a zero embedding should produce different
    Laplacians. The delta captures the allele-specific perturbation.
    """
    p = model_params
    node_seq = torch.randn(p["K"], p["N"], p["input_dim"])
    allele_emb = torch.randn(p["esm_dim"])
    ctx_L, full_L, delta_L = model.decompose_rewiring(node_seq, allele_emb)

    assert len(delta_L) == p["K"]
    # Deltas should not be all zero (FiLM is active)
    total_delta_norm = sum(d.norm().item() for d in delta_L)
    assert total_delta_norm > 0, "Delta Laplacians are all zero — FiLM is not modulating"


def test_decompose_rewiring_delta_zero_without_allele(model_no_allele, model_params):
    """Without allele conditioning, delta Laplacians should be zero.

    Theoretical: ZeroFiLM returns identity (gamma=1, beta=0), so the full
    model and the zero-embedding model produce identical Laplacians.
    """
    p = model_params
    node_seq = torch.randn(p["K"], p["N"], p["input_dim"])
    allele_emb = torch.randn(p["esm_dim"])
    _, _, delta_L = model_no_allele.decompose_rewiring(node_seq, allele_emb)

    total_delta_norm = sum(d.norm().item() for d in delta_L)
    assert total_delta_norm < 1e-4, (
        f"Delta norm {total_delta_norm} should be ~0 without allele conditioning"
    )


# ── Parameter Counting Tests ─────────────────────────────────────────────

def test_count_parameters(model):
    """count_parameters should return a dict with 'total' key matching actual count."""
    counts = model.count_parameters()
    assert "total" in counts
    actual = sum(p.numel() for p in model.parameters())
    assert counts["total"] == actual


# ── Factory Function Test ────────────────────────────────────────────────

def test_create_from_config(model_params):
    """Factory function should produce a valid model from config dict."""
    p = model_params
    config = {
        "model": {
            "stalk_dim": p["d"],
            "input_dim": p["input_dim"],
            "esm_dim": p["esm_dim"],
            "conditioning_dim": 64,
            "num_diffusion_steps": 2,
        }
    }
    edge_index = torch.randint(0, p["N"], (2, p["E"]))
    model = create_tsgnn_from_config(config, edge_index)
    assert isinstance(model, TSGNN)
