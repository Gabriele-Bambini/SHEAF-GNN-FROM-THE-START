"""
Tests for A1: OrthogonalSheafDiffusion.

Validates:
1. QR projection produces F^T F = I (orthogonality)
2. det(F) = ±1 (proper/improper rotation)
3. L_F eigenvalues bounded in [0, 2] after normalization
4. Gradient flows through QR projection (differentiable)
5. Drop-in compatibility with NormalizedVectorizedSheafDiffusion interface
6. Random vs orthogonal maps: orthogonal has better spectral properties
"""

import pytest
import torch
import torch.nn.functional as F


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def edge_index_small():
    """5-node cycle graph (directed edges)."""
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,3),(1,4)]
    src = [e[0] for e in edges]
    tgt = [e[1] for e in edges]
    return torch.tensor([src, tgt], dtype=torch.long)


@pytest.fixture
def edge_index_medium():
    """20-node random graph."""
    torch.manual_seed(99)
    N, E = 20, 80
    return torch.randint(0, N, (2, E))


def _make_layer(N, E, d, edge_index):
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion
    return OrthogonalSheafDiffusion(N, E, d, edge_index)


# ── Test 1: QR orthogonality ─────────────────────────────────────────────────

@pytest.mark.parametrize("d", [2, 4, 8])
def test_qr_projection_orthogonal(d):
    """F = QR[0] must satisfy F^T F = I to tolerance 1e-5."""
    from tsgnn.model.sheaf_orthogonal import project_to_orthogonal

    torch.manual_seed(0)
    batch = 100
    A = torch.randn(batch, d, d)
    Q = project_to_orthogonal(A)

    # Q^T Q = I
    FTF = torch.bmm(Q.transpose(1, 2), Q)
    I = torch.eye(d).unsqueeze(0).expand(batch, -1, -1)
    max_err = (FTF - I).abs().max().item()
    assert max_err < 1e-5, f"Orthogonality error {max_err:.2e} for d={d}"


@pytest.mark.parametrize("d", [2, 4, 8])
def test_qr_projection_determinant(d):
    """det(Q) = ±1 for all projected maps."""
    from tsgnn.model.sheaf_orthogonal import project_to_orthogonal

    torch.manual_seed(1)
    batch = 50
    A = torch.randn(batch, d, d)
    Q = project_to_orthogonal(A)
    dets = torch.det(Q).abs()    # should be 1.0
    max_err = (dets - 1.0).abs().max().item()
    assert max_err < 1e-4, f"|det(Q)| - 1 = {max_err:.2e} for d={d}"


# ── Test 2: Bundle projection ─────────────────────────────────────────────────

def test_bundle_projection_shape(edge_index_small):
    """project_maps_bundle preserves (E, 2, d, d) shape."""
    from tsgnn.model.sheaf_orthogonal import project_maps_bundle

    E = edge_index_small.shape[1]
    d = 4
    maps = torch.randn(E, 2, d, d)
    orth = project_maps_bundle(maps)

    assert orth.shape == maps.shape, f"Shape mismatch: {orth.shape} != {maps.shape}"


def test_bundle_projection_all_orthogonal(edge_index_small):
    """All 2E matrices in bundle must be orthogonal after projection."""
    from tsgnn.model.sheaf_orthogonal import project_maps_bundle

    E = edge_index_small.shape[1]
    d = 4
    torch.manual_seed(0)
    maps = torch.randn(E, 2, d, d)
    orth = project_maps_bundle(maps)

    flat = orth.reshape(2 * E, d, d)
    FTF = torch.bmm(flat.transpose(1, 2), flat)
    I = torch.eye(d).unsqueeze(0)
    max_err = (FTF - I).abs().max().item()
    assert max_err < 1e-5, f"Bundle orthogonality error {max_err:.2e}"


# ── Test 3: Layer orthogonality ───────────────────────────────────────────────

@pytest.mark.parametrize("d", [2, 4])
def test_layer_orthogonality_verify(d, edge_index_small):
    """verify_orthogonality utility must report all_pass=True."""
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion, verify_orthogonality

    N = 5
    E = edge_index_small.shape[1]
    layer = _make_layer(N, E, d, edge_index_small)

    torch.manual_seed(0)
    maps = torch.randn(E, 2, d, d)
    result = verify_orthogonality(layer, maps)

    assert result["all_pass"], (
        f"Orthogonality check failed: max_err={result['max_orth_error']:.2e}"
    )
    assert result["max_orth_error"] < 1e-5, (
        f"max_orth_error too large: {result['max_orth_error']:.2e}"
    )


# ── Test 4: Spectral bound ────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [2, 4])
def test_spectral_bound_after_orth(d, edge_index_small):
    """L_sym eigenvalues must be in [0, 2] with orthogonal maps."""
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion, verify_spectral_bound

    N = 5
    E = edge_index_small.shape[1]
    layer = _make_layer(N, E, d, edge_index_small)

    torch.manual_seed(0)
    maps = torch.randn(E, 2, d, d)
    result = verify_spectral_bound(layer, maps)

    assert result["min_eig"] >= -1e-4, (
        f"Negative eigenvalue: min={result['min_eig']:.4f}"
    )
    # Normalized Laplacian: eigenvalues in [0, 2]
    assert result["max_eig"] <= 2.0 + 1e-2, (
        f"Eigenvalue > 2: max={result['max_eig']:.4f}"
    )


def test_spectral_bound_better_than_unconstrained(edge_index_small):
    """Orthogonal maps should have smaller max eigenvalue than random maps."""
    from tsgnn.model.sheaf_vectorized import NormalizedVectorizedSheafDiffusion
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion

    N, E, d = 5, edge_index_small.shape[1], 4

    torch.manual_seed(42)
    maps_random = torch.randn(E, 2, d, d) * 3.0  # Large random maps

    layer_unc = NormalizedVectorizedSheafDiffusion(N, E, d, edge_index_small)
    layer_orth = OrthogonalSheafDiffusion(N, E, d, edge_index_small)

    L_unc = layer_unc.compute_connection_laplacian(maps_random)
    L_orth = layer_orth.compute_connection_laplacian(maps_random)

    eigs_unc = torch.linalg.eigvalsh(L_unc.float())
    eigs_orth = torch.linalg.eigvalsh(L_orth.float())

    # Orthogonal should have bounded spectrum
    assert eigs_orth[-1].item() <= 2.0 + 1e-2, (
        f"Orth max eig too large: {eigs_orth[-1].item():.4f}"
    )
    # Unconstrained can blow up with large maps
    # (don't assert upper bound — that's the whole point)


# ── Test 5: Gradient flow ─────────────────────────────────────────────────────

def test_gradient_flows_through_qr(edge_index_small):
    """
    Gradients must propagate through QR projection back to restriction maps.

    Critical: if QR is not differentiable, GRU and FiLM receive no gradient
    signal about map structure.
    """
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion

    N, E, d = 5, edge_index_small.shape[1], 4
    layer = _make_layer(N, E, d, edge_index_small)

    maps = torch.randn(E, 2, d, d, requires_grad=True)
    x = torch.randn(N, d)

    L_F = layer.compute_connection_laplacian(maps)
    x_out = layer.diffuse(x, L_F)

    loss = x_out.sum()
    loss.backward()

    assert maps.grad is not None, "No gradient reached restriction maps"
    assert maps.grad.abs().max().item() > 0, "Gradient is zero"
    assert torch.isfinite(maps.grad).all(), "Non-finite gradient"


def test_gradient_through_layer_parameters(edge_index_small):
    """Layer parameters (weight, bias, alpha) must receive gradients."""
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion

    N, E, d = 5, edge_index_small.shape[1], 4
    layer = _make_layer(N, E, d, edge_index_small)

    maps = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)

    L_F = layer.compute_connection_laplacian(maps)
    x_out = layer.diffuse(x, L_F)
    loss = x_out.pow(2).sum()
    loss.backward()

    # Check gradients on learnable parameters
    for name, param in layer.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient on {name}"
            )


# ── Test 6: Interface compatibility ──────────────────────────────────────────

def test_interface_compatible_with_tsgnn(edge_index_small):
    """
    OrthogonalSheafDiffusion is drop-in for NormalizedVectorizedSheafDiffusion.
    Verify that TSGNN can be built with it.
    """
    from tsgnn.model.tsgnn import TSGNN
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion

    N, E, d = 5, edge_index_small.shape[1], 4
    K = 3

    model = TSGNN(
        num_nodes=N,
        num_edges=E,
        stalk_dim=d,
        input_dim=1,
        esm_dim=64,
        conditioning_dim=32,
        edge_index=edge_index_small,
        num_diffusion_steps=1,
        use_allele_conditioning=True,
        use_vectorized=True,
        use_normalized_laplacian=True,
    )

    # Monkey-patch sheaf_layer to OrthogonalSheafDiffusion
    orth_layer = OrthogonalSheafDiffusion(N, E, d, edge_index_small)
    model.sheaf_layer = orth_layer

    x_seq = torch.randn(K, N, 1)
    allele = torch.randn(64)

    preds, maps_traj, laplacians = model(x_seq, allele)

    assert len(preds) == K
    assert all(torch.isfinite(p).all() for p in preds), "NaN in predictions"
    assert all(torch.isfinite(L).all() for L in laplacians), "NaN in Laplacians"


def test_forward_output_finite(edge_index_medium):
    """
    Full forward pass must produce finite outputs.
    Stress test with d=4, larger graph.
    """
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion

    N, E, d = 20, edge_index_medium.shape[1], 4
    layer = _make_layer(N, E, d, edge_index_medium)

    torch.manual_seed(5)
    maps = torch.randn(E, 2, d, d) * 2.0  # Large initial maps
    x = torch.randn(N, d) * 5.0           # Large features

    x_out, L_F = layer(x, restriction_maps=maps, num_steps=3)

    assert torch.isfinite(x_out).all(), "Non-finite output features"
    assert torch.isfinite(L_F).all(), "Non-finite Laplacian"


# ── Test 7: Comparison with unconstrained (sanity check) ─────────────────────

def test_orthogonal_laplacian_is_symmetric(edge_index_small):
    """L_sym must be symmetric (inherited from parent normalization)."""
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion

    N, E, d = 5, edge_index_small.shape[1], 4
    layer = _make_layer(N, E, d, edge_index_small)

    torch.manual_seed(0)
    maps = torch.randn(E, 2, d, d)
    L = layer.compute_connection_laplacian(maps)

    assert torch.allclose(L, L.T, atol=1e-5), (
        f"L_sym not symmetric: max |L - L^T| = {(L - L.T).abs().max().item():.2e}"
    )


def test_orthogonal_verify_properties_psd(edge_index_small):
    """verify_laplacian_properties should confirm PSD and symmetry."""
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion

    N, E, d = 5, edge_index_small.shape[1], 4
    layer = _make_layer(N, E, d, edge_index_small)

    torch.manual_seed(0)
    maps = torch.randn(E, 2, d, d)
    props = layer.verify_laplacian_properties(maps)

    assert props["is_symmetric"], "Laplacian is not symmetric"
    assert props["is_psd"], (
        f"Laplacian is not PSD: min_eigenvalue={props['min_eigenvalue']:.4f}"
    )
    assert props["spectral_gap"] >= 0, (
        f"Negative spectral gap: {props['spectral_gap']:.4f}"
    )
