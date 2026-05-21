"""
Reverse ablation: which feature in A1+A2+A3+B1+B2 hurts most?

For each feature X in {A1, A2, A3, B1, B2}, train TSGNN with all features
EXCEPT X enabled. Compare to full stack and pure baseline.

Configs (8 total):
  baseline:          no features
  full:              A1+A2+A3+B1+B2
  full-A1:           A2+A3+B1+B2 (orthogonal off, use_orthogonal_maps=False)
  full-A2:           A1+A3+B1+B2 (implicit off, use_implicit_diffusion=False)
  full-A3:           A1+A2+B1+B2 (predict_delta off)
  full-B1:           A1+A2+A3+B2 (nodal FiLM off, use global FiLM)
  full-B2:           A1+A2+A3+B1 (neural ODE off, GRU)

Reports BRCA r_delta + train_loss for each. Identifies hurters.

If full-X >> full, X is the hurter. If full-X <= full, X is fine.
"""
import sys, os
sys.path.insert(0, "/home/3393519/ts-gnn/src")
os.chdir("/home/3393519/ts-gnn")

import torch
import numpy as np
import yaml
import logging
logging.basicConfig(level=logging.WARNING)

from tsgnn.data.registry import load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors

print("=== Reverse Ablation: which feature hurts? ===")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_EPOCHS = 50
torch.manual_seed(SEED)

print("\n[1] Loading BRCA...")
ad_brca = load_dataset("brca_wu2021")
b_brca = adata_to_tsgnn_tensors(ad_brca, K=10, n_genes=500, k_neighbors=8, seed=SEED, pseudotime_method="dpt")
X_brca = b_brca["X_seq"].to(device)
ei_brca = b_brca["edge_index"].to(device)
K, N, _ = X_brca.shape
E = ei_brca.shape[1]
print(f"  BRCA: K={K}, N={N}, E={E}")

torch.manual_seed(SEED)
esm = torch.randn(1280, device=device) * 0.1


def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def eval_model(model, X_full, esm_vec):
    model.eval()
    with torch.no_grad():
        out = model(X_full[:-1], esm_vec)
        preds = out[0] if isinstance(out, tuple) else out
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=0)
    p_last = preds[-1].cpu().numpy().squeeze()
    x_last = X_full[-1].cpu().numpy().squeeze()
    x_prev = X_full[-2].cpu().numpy().squeeze()
    delta_pred = p_last - x_prev
    delta_true = x_last - x_prev
    return {
        "r_abs": pearson(p_last, x_last),
        "r_delta": pearson(delta_pred, delta_true),
    }


def train_model(model, X_full, esm_vec, n_epochs=N_EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    K = X_full.shape[0]
    val_k = max(1, K // 5)
    X_train = X_full[:-val_k - 1]
    Y_train = X_full[1:-val_k]
    model.train()
    for ep in range(n_epochs):
        opt.zero_grad()
        out = model(X_train, esm_vec)
        preds = out[0] if isinstance(out, tuple) else out
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=0)
        use_delta = getattr(model, "predict_delta", False)
        if use_delta:
            loss = ((preds - Y_train) ** 2).mean()
        else:
            loss = ((preds - X_train - (Y_train - X_train)) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.item()


from tsgnn.model.tsgnn import TSGNN


def run(name, **kwargs):
    print(f"\n[{name}] Training {N_EPOCHS} epochs...")
    torch.manual_seed(SEED)
    base = dict(
        num_nodes=N, num_edges=E,
        input_dim=1, esm_dim=1280, conditioning_dim=128,
        edge_index=ei_brca, num_diffusion_steps=3,
        use_allele_conditioning=True,
        use_vectorized=True,
        use_normalized_laplacian=True,
    )
    base.update(kwargs)
    model = TSGNN(**base).to(device)
    loss = train_model(model, X_brca, esm)
    res = eval_model(model, X_brca, esm)
    print(f"  loss={loss:.4f}  r_abs={res['r_abs']:.4f}  r_delta={res['r_delta']:.4f}")
    return {"name": name, "loss": loss, **res}


# Define configs
FULL = dict(
    stalk_dim=4,
    use_orthogonal_maps=True,
    use_implicit_diffusion=True,
    predict_delta=True,
    use_nodal_film=True,
    use_neural_ode=True,
    ode_hidden_dim=256, ode_solver="rk4",
)
BASELINE = dict(
    stalk_dim=8,
    use_orthogonal_maps=False,
    use_implicit_diffusion=False,
    predict_delta=False,
    use_nodal_film=False,
    use_neural_ode=False,
)


def with_off(feature):
    """Full stack with one feature disabled."""
    cfg = dict(FULL)
    cfg[feature] = False
    return cfg


configs = [
    ("baseline",  BASELINE),
    ("full",      FULL),
    ("full-A1",   with_off("use_orthogonal_maps")),
    ("full-A2",   with_off("use_implicit_diffusion")),
    ("full-A3",   with_off("predict_delta")),
    ("full-B1",   with_off("use_nodal_film")),
    ("full-B2",   with_off("use_neural_ode")),
]

results = [run(name, **kw) for name, kw in configs]

print("\n=== REVERSE ABLATION SUMMARY ===")
print(f"{'Config':<18} {'loss':>10} {'r_abs':>10} {'r_delta':>10}")
print("-" * 55)
for r in results:
    print(f"  {r['name']:<16} {r['loss']:>10.4f} {r['r_abs']:>10.4f} {r['r_delta']:>10.4f}")

# Highlight hurters: full-X >> full means X hurts
full = next(r for r in results if r["name"] == "full")
print(f"\nFull-stack r_delta = {full['r_delta']:.4f}")
print("\nFeature impact (full-X vs full):")
for r in results:
    if r["name"].startswith("full-"):
        delta = r["r_delta"] - full["r_delta"]
        sign = "+" if delta >= 0 else ""
        verdict = "HURTS (remove)" if delta > 0.05 else ("HELPS" if delta < -0.05 else "neutral")
        print(f"  {r['name']:<14} delta={sign}{delta:>7.4f}   {verdict}")

best = max(results, key=lambda r: r["r_delta"])
print(f"\nBest config: {best['name']} (r_delta={best['r_delta']:.4f})")
print(f"P1 threshold (0.35): {'PASS' if best['r_delta'] > 0.35 else 'FAIL'}")
