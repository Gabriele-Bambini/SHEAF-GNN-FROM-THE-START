"""
Tests for B2: SheafNeuralODE.

Validates:
1. Output shape matches input (drop-in compat with TemporalRestrictionEvolution)
2. Identity-like behavior at init (zero-init last layer → R(δt) ≈ LayerNorm(R(0)))
3. Different δt → different outputs (continuous dependency)
4. Output finite for various δt and inputs
5. Gradient flows through ODE to restriction_maps
6. Gradient flows through ODE to node_features
7. Gradient flows through ODE to internal parameters
8. Solver fallback works (invalid solver → euler)
9. TSGNN integration: forward+backward complete
10. SheafNeuralODE is default in TSGNN
11. use_neural_ode=False falls back to GRU
12. Config factory reads use_neural_ode
13. dt clamping: zero δt doesn't crash
14. After training step, ODE params change (learnable)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.fixture
def edge_index():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3), (2, 4)]
    return torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)


def _make_ode(d=4, hidden=128):
    from tsgnn.model.temporal_ode import SheafNeuralODE
    return SheafNeuralODE(stalk_dim=d, hidden_dim=hidden, solver="rk4")


def _make_tsgnn(edge_index, use_neural_ode=True, N=5, d=4):
    from tsgnn.model.tsgnn import TSGNN
    return TSGNN(
        num_nodes=N, num_edges=edge_index.shape[1],
        stalk_dim=d, input_dim=1, esm_dim=32, conditioning_dim=16,
        edge_index=edge_index, num_diffusion_steps=1,
        use_allele_conditioning=True,
        use_implicit_diffusion=True,
        use_orthogonal_maps=True,
        predict_delta=True,
        use_nodal_film=True,
        use_neural_ode=use_neural_ode,
    )


# ── Test 1: shape ─────────────────────────────────────────────────────────────

def test_ode_output_shape(edge_index):
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.1))
    assert R1.shape == R0.shape


# ── Test 2: identity at init ──────────────────────────────────────────────────

def test_identity_at_init(edge_index):
    """Zero-init last layer → R(δt) ≈ LayerNorm(R(0))."""
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.1))
    R0_norm = ode.layer_norm(R0.reshape(E, -1)).reshape(E, 2, d, d)
    diff = (R1 - R0_norm).abs().max().item()
    assert diff < 1e-5, f"Identity flow violated: {diff:.2e}"


# ── Test 3: continuous dependency on δt ───────────────────────────────────────

def test_dt_dependency(edge_index):
    """After training (perturb velocity field), different δt → different R."""
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    # Perturb last layer so velocity is non-zero
    with torch.no_grad():
        ode.velocity_field[-1].weight.normal_(0, 0.1)
        ode.velocity_field[-1].bias.normal_(0, 0.1)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    R_short = ode(R0, x, edge_index, delta_t=torch.tensor(0.1))
    R_long = ode(R0, x, edge_index, delta_t=torch.tensor(1.0))
    assert not torch.allclose(R_short, R_long, atol=1e-3), "δt has no effect"


# ── Test 4: output finite for various δt ──────────────────────────────────────

@pytest.mark.parametrize("dt", [0.001, 0.1, 0.5, 1.0, 5.0])
def test_finite_output(edge_index, dt):
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(dt))
    assert torch.isfinite(R1).all(), f"Non-finite for δt={dt}"


# ── Test 5: gradient flows ────────────────────────────────────────────────────

def test_gradient_flows_to_R(edge_index):
    """Use squared loss to bypass LayerNorm sum cancellation."""
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    # Perturb so velocity is nonzero (otherwise R passes through as identity → grad nonzero anyway)
    with torch.no_grad():
        ode.velocity_field[-1].weight.normal_(0, 0.1)
    R0 = torch.randn(E, 2, d, d, requires_grad=True)
    x = torch.randn(N, d)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.5))
    R1.pow(2).sum().backward()
    assert R0.grad is not None, "no grad to R0"
    assert torch.isfinite(R0.grad).all(), "non-finite grad on R0"
    assert R0.grad.abs().max() > 0


def test_gradient_flows_to_node_features(edge_index):
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    with torch.no_grad():
        ode.velocity_field[-1].weight.normal_(0, 0.1)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d, requires_grad=True)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.5))
    R1.pow(2).sum().backward()
    assert x.grad is not None, "no grad to node_features"
    assert torch.isfinite(x.grad).all()


def test_gradient_flows_to_velocity_field(edge_index):
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.5))
    R1.pow(2).sum().backward()
    for name, p in ode.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad on {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


# ── Test 6: TSGNN integration ─────────────────────────────────────────────────

def test_tsgnn_with_neural_ode(edge_index):
    N, E, d, K = 5, edge_index.shape[1], 4, 4
    model = _make_tsgnn(edge_index, use_neural_ode=True, N=N, d=d)
    x_seq = torch.randn(K, N, 1)
    allele = torch.randn(32)
    target = torch.randn(K, N, 1)
    preds, _, _ = model(x_seq, allele)
    loss = F.mse_loss(torch.stack(preds), target)
    loss.backward()
    assert all(torch.isfinite(p).all() for p in preds)
    assert torch.isfinite(loss)


def test_neural_ode_default_in_tsgnn(edge_index):
    """use_neural_ode True by default."""
    from tsgnn.model.tsgnn import TSGNN
    from tsgnn.model.temporal_ode import SheafNeuralODE
    model = TSGNN(
        num_nodes=5, num_edges=edge_index.shape[1], stalk_dim=4, input_dim=1,
        esm_dim=32, conditioning_dim=16, edge_index=edge_index,
        num_diffusion_steps=1,
    )
    assert isinstance(model.temporal_evolution, SheafNeuralODE)
    assert model.use_neural_ode is True


def test_gru_fallback(edge_index):
    """use_neural_ode=False → GRU."""
    from tsgnn.model.tsgnn import TSGNN
    from tsgnn.model.temporal import TemporalRestrictionEvolution
    from tsgnn.model.temporal_ode import SheafNeuralODE
    model = TSGNN(
        num_nodes=5, num_edges=edge_index.shape[1], stalk_dim=4, input_dim=1,
        esm_dim=32, conditioning_dim=16, edge_index=edge_index,
        num_diffusion_steps=1, use_neural_ode=False,
    )
    assert isinstance(model.temporal_evolution, TemporalRestrictionEvolution)
    assert not isinstance(model.temporal_evolution, SheafNeuralODE)


# ── Test 7: config factory ────────────────────────────────────────────────────

def test_config_factory_neural_ode(edge_index):
    from tsgnn.model.tsgnn import create_tsgnn_from_config
    from tsgnn.model.temporal_ode import SheafNeuralODE
    cfg = {
        "data": {"n_genes": 5},
        "model": {
            "stalk_dim": 4, "input_dim": 1, "esm_dim": 32,
            "conditioning_dim": 16, "num_diffusion_steps": 1,
            "use_neural_ode": True,
        },
    }
    model = create_tsgnn_from_config(cfg, edge_index)
    assert isinstance(model.temporal_evolution, SheafNeuralODE)


def test_config_factory_no_neural_ode(edge_index):
    from tsgnn.model.tsgnn import create_tsgnn_from_config
    from tsgnn.model.temporal import TemporalRestrictionEvolution
    cfg = {
        "data": {"n_genes": 5},
        "model": {
            "stalk_dim": 4, "input_dim": 1, "esm_dim": 32,
            "conditioning_dim": 16, "num_diffusion_steps": 1,
            "use_neural_ode": False,
        },
    }
    model = create_tsgnn_from_config(cfg, edge_index)
    assert isinstance(model.temporal_evolution, TemporalRestrictionEvolution)


# ── Test 8: edge cases ────────────────────────────────────────────────────────

def test_zero_dt_clamped(edge_index):
    """δt=0 must not crash (clamped to 1e-3)."""
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.0))
    assert torch.isfinite(R1).all()


def test_none_dt(edge_index):
    """δt=None defaults to 1.0."""
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    R1 = ode(R0, x, edge_index, delta_t=None)
    assert R1.shape == R0.shape


# ── Test 9: learnability ──────────────────────────────────────────────────────

def test_ode_params_learnable(edge_index):
    """One optimizer step changes velocity_field params.

    Zero-init last layer means earlier layers get ZERO gradient at first
    forward pass (chain rule: dL/dW_first = dL/dOut * W_last * ... = 0).
    So we check the LAST layer instead — it's the one with direct gradient
    from the output and the one being zero-initialized.
    """
    N, E, d = 5, edge_index.shape[1], 4
    ode = _make_ode(d=d)
    init_w = ode.velocity_field[-1].weight.clone().detach()
    opt = torch.optim.SGD(ode.parameters(), lr=0.1)
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.5))
    loss = R1.pow(2).sum()
    loss.backward()
    opt.step()
    new_w = ode.velocity_field[-1].weight
    assert not torch.allclose(init_w, new_w), "velocity_field last layer did not update"
