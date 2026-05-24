"""
Tests for B1: NodalAlleleFiLM + LaplacianPE.

Validates:
1. LaplacianPE: shape, finite values, different topology → different PE
2. LaplacianPE: cached after first call (same tensor returned)
3. NodalAlleleFiLM: output shape matches input
4. NodalAlleleFiLM: different alleles → different outputs (allele sensitivity)
5. NodalAlleleFiLM: same allele → same output (deterministic)
6. Gradient flows through NodalAlleleFiLM to allele embedding
7. Gradient flows through NodalAlleleFiLM to gru_output
8. TSGNN with use_nodal_film=True completes forward+backward
9. NodalAlleleFiLM is default in TSGNN (use_nodal_film=True by default)
10. use_nodal_film=False falls back to AlleleFiLM
11. Output finite for edge cases (zero allele, large maps)
12. Config factory reads use_nodal_film
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.fixture
def edge_index():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3), (2, 4)]
    src = [e[0] for e in edges]
    tgt = [e[1] for e in edges]
    return torch.tensor([src, tgt], dtype=torch.long)


@pytest.fixture
def edge_index_large():
    torch.manual_seed(42)
    return torch.randint(0, 20, (2, 80))


def _make_nodal_film(N=5, E=8, d=4, esm_dim=32):
    from tsgnn.model.film_nodal import NodalAlleleFiLM
    return NodalAlleleFiLM(
        num_nodes=N,
        esm_dim=esm_dim,
        conditioning_dim=16,
        modulation_dim=2 * d * d,
    )


def _make_tsgnn(edge_index, use_nodal_film=True, N=5, d=4, K=4):
    from tsgnn.model.tsgnn import TSGNN
    return TSGNN(
        num_nodes=N, num_edges=edge_index.shape[1],
        stalk_dim=d, input_dim=1, esm_dim=32, conditioning_dim=16,
        edge_index=edge_index, num_diffusion_steps=1,
        use_allele_conditioning=True,
        use_implicit_diffusion=True,
        use_orthogonal_maps=True,
        predict_delta=True,
        use_nodal_film=use_nodal_film,
    )


# ── Test 1: LaplacianPE shape and finiteness ──────────────────────────────────

def test_laplacian_pe_shape(edge_index):
    from tsgnn.model.film_nodal import LaplacianPE
    N, k = 5, 4
    pe_layer = LaplacianPE(N, num_eigs=k)
    pe = pe_layer(edge_index)
    assert pe.shape == (N, k), f"Expected ({N}, {k}), got {pe.shape}"
    assert torch.isfinite(pe).all(), "Non-finite PE values"


def test_laplacian_pe_different_topologies():
    """Different graph topology → different PE (not identical)."""
    from tsgnn.model.film_nodal import LaplacianPE
    N = 10
    torch.manual_seed(0)
    ei1 = torch.randint(0, N, (2, 20))
    ei2 = torch.randint(0, N, (2, 20))

    pe1 = LaplacianPE(N, num_eigs=4)(ei1)
    pe2 = LaplacianPE(N, num_eigs=4)(ei2)
    assert not torch.allclose(pe1, pe2, atol=1e-4), "Different topologies gave same PE"


def test_laplacian_pe_cached(edge_index):
    """PE cached: second call returns same tensor without recomputing."""
    from tsgnn.model.film_nodal import LaplacianPE
    pe_layer = LaplacianPE(5, num_eigs=4)
    pe1 = pe_layer(edge_index)
    pe2 = pe_layer()  # no edge_index — should use cache
    assert torch.allclose(pe1, pe2), "Cached PE differs from first call"


# ── Test 2: NodalAlleleFiLM output shape ─────────────────────────────────────

def test_nodal_film_output_shape(edge_index):
    N, E, d = 5, edge_index.shape[1], 4
    film = _make_nodal_film(N, E, d)
    esm = torch.randn(32)
    gru = torch.randn(E, 2 * d * d)
    out = film(esm, gru, edge_index)
    assert out.shape == (E, 2 * d * d), f"Expected ({E}, {2*d*d}), got {out.shape}"


# ── Test 3: Allele sensitivity ────────────────────────────────────────────────

def test_nodal_film_allele_sensitivity(edge_index):
    """Different allele embeddings → different outputs."""
    N, E, d = 5, edge_index.shape[1], 4
    film = _make_nodal_film(N, E, d)
    gru = torch.randn(E, 2 * d * d)
    out1 = film(torch.randn(32), gru, edge_index)
    out2 = film(torch.randn(32), gru, edge_index)
    assert not torch.allclose(out1, out2, atol=1e-4), "Different alleles gave identical output"


def test_nodal_film_deterministic(edge_index):
    """Same allele + same gru → same output (eval mode, no Dropout noise)."""
    N, E, d = 5, edge_index.shape[1], 4
    film = _make_nodal_film(N, E, d)
    film.eval()  # disable Dropout
    with torch.no_grad():
        esm = torch.randn(32)
        gru = torch.randn(E, 2 * d * d)
        out1 = film(esm, gru, edge_index)
        out2 = film(esm, gru, edge_index)
    assert torch.allclose(out1, out2), "Same inputs gave different outputs (eval mode)"


# ── Test 4: Gradient flows ────────────────────────────────────────────────────

def test_gradient_through_allele(edge_index):
    """Gradient must flow from output back to ESM embedding."""
    N, E, d = 5, edge_index.shape[1], 4
    film = _make_nodal_film(N, E, d)
    esm = torch.randn(32, requires_grad=True)
    gru = torch.randn(E, 2 * d * d)
    out = film(esm, gru, edge_index)
    out.sum().backward()
    assert esm.grad is not None, "No gradient on ESM embedding"
    assert torch.isfinite(esm.grad).all(), "Non-finite gradient on ESM"
    assert esm.grad.abs().max() > 0, "Zero gradient on ESM"


def test_gradient_through_gru(edge_index):
    """Gradient must flow from output back to gru_output."""
    N, E, d = 5, edge_index.shape[1], 4
    film = _make_nodal_film(N, E, d)
    esm = torch.randn(32)
    gru = torch.randn(E, 2 * d * d, requires_grad=True)
    out = film(esm, gru, edge_index)
    out.sum().backward()
    assert gru.grad is not None, "No gradient on gru_output"
    assert torch.isfinite(gru.grad).all(), "Non-finite gradient on gru_output"


# ── Test 5: TSGNN integration ─────────────────────────────────────────────────

def test_tsgnn_with_nodal_film(edge_index):
    """TSGNN with use_nodal_film=True must complete forward+backward."""
    N, E, d, K = 5, edge_index.shape[1], 4, 4
    model = _make_tsgnn(edge_index, use_nodal_film=True, N=N, d=d, K=K)

    x_seq = torch.randn(K, N, 1)
    allele = torch.randn(32)
    target = torch.randn(K, N, 1)

    preds, _, _ = model(x_seq, allele)
    preds_t = torch.stack(preds)
    loss = F.mse_loss(preds_t, target)
    loss.backward()

    assert all(torch.isfinite(p).all() for p in preds), "NaN in predictions"
    assert torch.isfinite(loss), "NaN loss"


def test_nodal_film_default_in_tsgnn(edge_index):
    """use_nodal_film should be True by default."""
    from tsgnn.model.tsgnn import TSGNN
    from tsgnn.model.film_nodal import NodalAlleleFiLM

    N, E, d = 5, edge_index.shape[1], 4
    model = TSGNN(
        num_nodes=N, num_edges=E,
        stalk_dim=d, input_dim=1, esm_dim=32, conditioning_dim=16,
        edge_index=edge_index, num_diffusion_steps=1,
    )
    assert isinstance(model.allele_film, NodalAlleleFiLM), (
        f"Default allele_film should be NodalAlleleFiLM, got {type(model.allele_film).__name__}"
    )
    assert model.use_nodal_film is True


def test_global_film_fallback(edge_index):
    """use_nodal_film=False → AlleleFiLM (not nodal)."""
    from tsgnn.model.tsgnn import TSGNN
    from tsgnn.model.film import AlleleFiLM
    from tsgnn.model.film_nodal import NodalAlleleFiLM

    N, E, d = 5, edge_index.shape[1], 4
    model = TSGNN(
        num_nodes=N, num_edges=E,
        stalk_dim=d, input_dim=1, esm_dim=32, conditioning_dim=16,
        edge_index=edge_index, num_diffusion_steps=1,
        use_nodal_film=False,
    )
    assert isinstance(model.allele_film, AlleleFiLM), "Should be AlleleFiLM"
    assert not isinstance(model.allele_film, NodalAlleleFiLM), "Should NOT be NodalAlleleFiLM"


# ── Test 6: Output finite for edge cases ─────────────────────────────────────

def test_zero_allele_finite(edge_index):
    """Zero allele embedding must not produce NaN."""
    N, E, d = 5, edge_index.shape[1], 4
    film = _make_nodal_film(N, E, d)
    out = film(torch.zeros(32), torch.randn(E, 2 * d * d), edge_index)
    assert torch.isfinite(out).all(), f"Non-finite output for zero allele"


def test_large_gru_finite(edge_index):
    """Large GRU values must not overflow."""
    N, E, d = 5, edge_index.shape[1], 4
    film = _make_nodal_film(N, E, d)
    out = film(torch.randn(32), torch.randn(E, 2 * d * d) * 100, edge_index)
    assert torch.isfinite(out).all(), f"Non-finite output for large GRU"


# ── Test 7: Config factory ────────────────────────────────────────────────────

def test_config_factory_nodal_film(edge_index):
    """Factory reads use_nodal_film from config."""
    from tsgnn.model.tsgnn import create_tsgnn_from_config
    from tsgnn.model.film_nodal import NodalAlleleFiLM

    cfg = {
        "data": {"n_genes": 5},
        "model": {
            "stalk_dim": 4, "input_dim": 1, "esm_dim": 32,
            "conditioning_dim": 16, "num_diffusion_steps": 1,
            "use_nodal_film": True,
        },
    }
    model = create_tsgnn_from_config(cfg, edge_index)
    assert isinstance(model.allele_film, NodalAlleleFiLM)


def test_config_factory_no_nodal_film(edge_index):
    """use_nodal_film=False in config → AlleleFiLM."""
    from tsgnn.model.tsgnn import create_tsgnn_from_config
    from tsgnn.model.film import AlleleFiLM

    cfg = {
        "data": {"n_genes": 5},
        "model": {
            "stalk_dim": 4, "input_dim": 1, "esm_dim": 32,
            "conditioning_dim": 16, "num_diffusion_steps": 1,
            "use_nodal_film": False,
        },
    }
    model = create_tsgnn_from_config(cfg, edge_index)
    assert isinstance(model.allele_film, AlleleFiLM)
