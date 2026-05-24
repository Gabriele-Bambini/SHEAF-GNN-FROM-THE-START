"""
Tests for A2: ImplicitOrthogonalSheafDiffusion (Crank-Nicolson).

Validates:
1. Energy monotonicity: E(x_next) ≤ E(x_curr)  (key property of CN)
2. Stability for large τ (where explicit Euler would blow up)
3. CG convergence: residual ‖Ax - b‖ < tol
4. Gradient flows through CG solve (differentiable)
5. τ is a learnable parameter
6. Interface compatibility with TSGNN
7. Output finite for stress inputs
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.fixture
def edge_index():
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2),(1,3),(2,4)]
    src = [e[0] for e in edges]
    tgt = [e[1] for e in edges]
    return torch.tensor([src, tgt], dtype=torch.long)


@pytest.fixture
def edge_index_medium():
    torch.manual_seed(7)
    return torch.randint(0, 20, (2, 80))


def _make_layer(N, E, d, edge_index, tau_init=0.2):
    from tsgnn.model.diffusion_implicit import ImplicitOrthogonalSheafDiffusion
    return ImplicitOrthogonalSheafDiffusion(N, E, d, edge_index, tau_init=tau_init)


# ── Test 1: Energy monotonicity ───────────────────────────────────────────────

@pytest.mark.parametrize("tau_init", [0.1, 0.3, 0.5, 1.0])
def test_energy_monotone(edge_index, tau_init):
    """
    CN diffusion must reduce Dirichlet energy: E(x_next) ≤ E(x_curr).
    This is the fundamental property of parabolic diffusion equations.
    """
    N, E, d = 5, edge_index.shape[1], 4
    torch.manual_seed(0)
    maps = torch.randn(E, 2, d, d)
    x = torch.randn(N, d) * 2.0

    layer = _make_layer(N, E, d, edge_index, tau_init=tau_init)

    with torch.no_grad():
        L_F = layer.compute_connection_laplacian(maps)
        x_next = layer.diffuse(x, L_F)

    E_before = 0.5 * (x.reshape(-1) @ L_F @ x.reshape(-1)).item()
    E_after  = 0.5 * (x_next.reshape(-1) @ L_F @ x_next.reshape(-1)).item()

    assert E_after <= E_before + 1e-3, (
        f"Energy increased for τ={tau_init}: {E_before:.4f} → {E_after:.4f}"
    )


# ── Test 2: Stability for large τ (explicit Euler would diverge) ──────────────

def test_stability_large_tau(edge_index):
    """
    Explicit Euler blows up for large τ. CN must remain bounded.
    Test with τ=1.0 (clamped max) and large random L_F.
    """
    N, E, d = 5, edge_index.shape[1], 4
    torch.manual_seed(1)
    maps = torch.randn(E, 2, d, d) * 3.0  # large maps → large L_F

    layer = _make_layer(N, E, d, edge_index, tau_init=1.0)

    with torch.no_grad():
        L_F = layer.compute_connection_laplacian(maps)
        x = torch.randn(N, d) * 10.0   # large features

        x_next = layer.diffuse(x, L_F)

    assert torch.isfinite(x_next).all(), "Non-finite output for large τ"
    # CN should contract or at least not explode
    # (clamping at ±50 is the last line of defense, but CN should not need it)
    assert x_next.abs().max().item() <= 50.1, (
        f"Output too large: {x_next.abs().max().item():.2f}"
    )


# ── Test 3: CG convergence quality ────────────────────────────────────────────

def test_cg_residual_small(edge_index):
    """
    After CG, verify ‖A x_next - b‖ is small (actual solve quality).
    """
    from tsgnn.model.diffusion_implicit import conjugate_gradient

    N, E, d = 5, edge_index.shape[1], 4
    Nd = N * d
    torch.manual_seed(2)

    layer = _make_layer(N, E, d, edge_index, tau_init=0.2)
    maps = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)

    with torch.no_grad():
        L_F = layer.compute_connection_laplacian(maps)
        tau = layer.tau
        I = torch.eye(Nd)
        A = I + (tau * 0.5) * L_F
        b = x.reshape(Nd) - (tau * 0.5) * (L_F @ x.reshape(Nd))

        x_sol = conjugate_gradient(A, b, x0=x.reshape(Nd), max_iter=30, tol=1e-6)
        residual = (A @ x_sol - b).norm().item()

    assert residual < 1e-3, f"CG residual too large: {residual:.2e}"


# ── Test 4: Gradient flows through CG ─────────────────────────────────────────

def test_gradient_flows_through_cg(edge_index):
    """
    Autograd must differentiate through CG (all tensor ops tracked).
    Gradient w.r.t. restriction maps must be finite and non-zero.
    """
    N, E, d = 5, edge_index.shape[1], 4
    layer = _make_layer(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d, requires_grad=True)
    x = torch.randn(N, d)

    L_F = layer.compute_connection_laplacian(maps)
    x_next = layer.diffuse(x, L_F)
    loss = x_next.pow(2).sum()
    loss.backward()

    assert maps.grad is not None, "No gradient through CG"
    assert maps.grad.abs().max().item() > 0, "Zero gradient through CG"
    assert torch.isfinite(maps.grad).all(), "Non-finite gradient through CG"


def test_tau_gradient(edge_index):
    """τ (log_tau) must receive a gradient."""
    N, E, d = 5, edge_index.shape[1], 4
    layer = _make_layer(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)

    L_F = layer.compute_connection_laplacian(maps)
    x_next = layer.diffuse(x, L_F)
    x_next.pow(2).sum().backward()

    assert layer.log_tau.grad is not None, "No gradient on log_tau"
    assert torch.isfinite(layer.log_tau.grad).all(), "Non-finite gradient on log_tau"


# ── Test 5: τ is learnable ────────────────────────────────────────────────────

def test_tau_learnable(edge_index):
    """τ should change after one optimizer step."""
    N, E, d = 5, edge_index.shape[1], 4
    layer = _make_layer(N, E, d, edge_index, tau_init=0.2)
    tau_before = layer.tau.item()

    opt = torch.optim.SGD([layer.log_tau], lr=0.1)
    maps = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    L_F = layer.compute_connection_laplacian(maps)
    x_next = layer.diffuse(x, L_F)
    loss = x_next.pow(2).sum()
    loss.backward()
    opt.step()

    tau_after = layer.tau.item()
    assert abs(tau_after - tau_before) > 1e-6, "τ did not update"


# ── Test 6: Interface compatible with TSGNN ───────────────────────────────────

def test_tsgnn_with_implicit_diffusion(edge_index):
    """TSGNN with use_implicit_diffusion=True must complete forward+backward."""
    from tsgnn.model.tsgnn import TSGNN

    N, E, d, K = 5, edge_index.shape[1], 4, 4
    model = TSGNN(
        num_nodes=N, num_edges=E,
        stalk_dim=d, input_dim=1, esm_dim=32, conditioning_dim=16,
        edge_index=edge_index, num_diffusion_steps=1,
        use_allele_conditioning=True,
        use_implicit_diffusion=True,   # A2
        use_orthogonal_maps=True,      # A1 (already in ImplicitOrthogonal)
        predict_delta=True,            # A3
    )

    x_seq = torch.randn(K, N, 1)
    allele = torch.randn(32)
    target = torch.randn(K, N, 1)

    preds, _, _ = model(x_seq, allele)
    preds_t = torch.stack(preds)
    loss = F.mse_loss(preds_t, target)
    loss.backward()

    assert all(torch.isfinite(p).all() for p in preds), "NaN in predictions"
    assert torch.isfinite(loss), "NaN loss"

    for name, param in model.named_parameters():
        if "sheaf_layer.restriction_maps" in name:
            continue  # unused fallback in TSGNN
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"


def test_implicit_diffusion_default_in_tsgnn(edge_index):
    """use_implicit_diffusion should be True by default."""
    from tsgnn.model.tsgnn import TSGNN
    from tsgnn.model.diffusion_implicit import ImplicitOrthogonalSheafDiffusion

    N, E, d = 5, edge_index.shape[1], 4
    model = TSGNN(
        num_nodes=N, num_edges=E,
        stalk_dim=d, input_dim=1, esm_dim=32, conditioning_dim=16,
        edge_index=edge_index, num_diffusion_steps=1,
    )

    assert isinstance(model.sheaf_layer, ImplicitOrthogonalSheafDiffusion), (
        f"Default sheaf_layer should be ImplicitOrthogonalSheafDiffusion, "
        f"got {type(model.sheaf_layer).__name__}"
    )
    assert model.use_implicit_diffusion is True


# ── Test 7: Output finite for stress inputs ───────────────────────────────────

@pytest.mark.parametrize("scale", [0.01, 1.0, 10.0, 50.0])
def test_output_finite_stress(edge_index, scale):
    """Layer must produce finite output for a wide range of input scales."""
    N, E, d = 5, edge_index.shape[1], 4
    torch.manual_seed(scale)
    layer = _make_layer(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d) * scale
    x = torch.randn(N, d) * scale

    with torch.no_grad():
        x_out, L_F = layer(x, restriction_maps=maps, num_steps=1)

    assert torch.isfinite(x_out).all(), (
        f"Non-finite output at scale={scale}: max={x_out.abs().max().item():.2e}"
    )


# ── Test 8: create_tsgnn_from_config ─────────────────────────────────────────

def test_config_factory_implicit(edge_index):
    """Factory function must read use_implicit_diffusion from config."""
    from tsgnn.model.tsgnn import create_tsgnn_from_config
    from tsgnn.model.diffusion_implicit import ImplicitOrthogonalSheafDiffusion

    cfg = {
        "data": {"n_genes": 5},
        "model": {
            "stalk_dim": 4,
            "input_dim": 1,
            "esm_dim": 32,
            "conditioning_dim": 16,
            "num_diffusion_steps": 1,
            "use_implicit_diffusion": True,
        },
    }
    model = create_tsgnn_from_config(cfg, edge_index)
    assert isinstance(model.sheaf_layer, ImplicitOrthogonalSheafDiffusion)


def test_config_factory_no_implicit(edge_index):
    """use_implicit_diffusion=False should use OrthogonalSheafDiffusion."""
    from tsgnn.model.tsgnn import create_tsgnn_from_config
    from tsgnn.model.sheaf_orthogonal import OrthogonalSheafDiffusion
    from tsgnn.model.diffusion_implicit import ImplicitOrthogonalSheafDiffusion

    cfg = {
        "data": {"n_genes": 5},
        "model": {
            "stalk_dim": 4,
            "input_dim": 1,
            "esm_dim": 32,
            "conditioning_dim": 16,
            "num_diffusion_steps": 1,
            "use_implicit_diffusion": False,
            "use_orthogonal_maps": True,
        },
    }
    model = create_tsgnn_from_config(cfg, edge_index)
    # Should be OrthogonalSheafDiffusion but NOT the implicit variant
    assert isinstance(model.sheaf_layer, OrthogonalSheafDiffusion)
    assert not isinstance(model.sheaf_layer, ImplicitOrthogonalSheafDiffusion)
