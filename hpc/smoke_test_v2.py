"""TSGNN v2 smoke test — runs locally without HPC/data."""
import sys
sys.path.insert(0, 'src')
import torch

errors = []

# ── Test 1: Cayley orthogonality ─────────────────────────────────────────────
from tsgnn.model.cayley_orthogonal import verify_cayley_orthogonality, CayleyOrthogonalSheafDiffusion
for d in [4, 6]:
    r = verify_cayley_orthogonality(d=d)
    status = 'PASS' if r['all_pass'] else 'FAIL'
    print(f"Cayley d={d} [{status}] max_err={r['max_orth_error']:.2e}")
    if not r['all_pass']:
        errors.append(f"Cayley d={d} failed")

# ── Test 2: EdgeWeightNet gradient flow ESM→weights ──────────────────────────
from tsgnn.model.edge_weight_net import EdgeWeightNet
N, E, d = 20, 60, 6
edge_index = torch.randint(0, N, (2, E))
x   = torch.randn(N, d)
esm = torch.randn(128, requires_grad=True)
net = EdgeWeightNet(stalk_dim=d, esm_latent_dim=128)
w   = net(x, esm, edge_index)
w.sum().backward()
finite = torch.isfinite(esm.grad).all().item()
print(f"EdgeWeightNet ESM grad: max={esm.grad.abs().max():.4e} finite={finite}")
if not finite:
    errors.append("EdgeWeightNet gradient not finite")

# ── Test 3: Full TSGNN v2 forward ────────────────────────────────────────────
from tsgnn.model.tsgnn import TSGNN
model = TSGNN(
    num_nodes=N, num_edges=E, stalk_dim=d, input_dim=10,
    esm_dim=1280, conditioning_dim=128, edge_index=edge_index,
    use_cayley_v2=True, use_neural_ode=False,
)
K   = 5
seq = torch.randn(K, N, 10)
esm_full = torch.randn(1280)
preds, maps, laps = model(seq, esm_full)
ok_shape  = preds[0].shape == (N, 10)
ok_finite = torch.isfinite(preds[0]).all().item()
ok_sym    = torch.allclose(laps[0], laps[0].T, atol=1e-4)
print(f"TSGNN v2 forward: K={len(preds)} shape_ok={ok_shape} finite={ok_finite} sym_L={ok_sym}")
if not (ok_shape and ok_finite):
    errors.append("TSGNN forward failed")

# ── Test 4: Gradient through full model (ESM→weights→L→pred) ─────────────────
esm_grad = torch.randn(1280, requires_grad=True)
preds_g, _, laps_g = model(seq, esm_grad)
loss = sum(p.sum() for p in preds_g)
loss.backward()
g_finite = torch.isfinite(esm_grad.grad).all().item()
print(f"Full model ESM gradient: finite={g_finite}")
if not g_finite:
    errors.append("Full model gradient not finite")

# ── Test 5: Residualized metric ───────────────────────────────────────────────
from tsgnn.evaluation.metrics import residualized_delta_corr
pd_t = torch.randn(50, 10)
td_t = torch.randn(50, 10)
res  = residualized_delta_corr(pd_t, td_t)
print(f"r_delta_perp metric: r_raw={res['r_delta']:.4f} r_perp={res['r_delta_perp']:.4f} interp={res['interpretation']}")

# ── Summary ───────────────────────────────────────────────────────────────────
if errors:
    print(f"\nFAILED: {errors}")
    sys.exit(1)
else:
    print("\nALL SMOKE TESTS PASSED ✓")
    sys.exit(0)
