"""
Final probe: A1+A2+B2 (chosen) vs baseline, 3 seeds.

After reverse ablation 482916 identified A3+B1 as harmful,
this script confirms A1+A2+B2 beats baseline with statistical
significance (3 seeds, paired r_delta comparison).

Reports:
  - per-seed r_delta for both configs
  - mean ± std
  - paired delta + sign test
"""
import sys, os
sys.path.insert(0, "/home/3393519/ts-gnn/src")
os.chdir("/home/3393519/ts-gnn")

import torch
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

from tsgnn.data.registry import load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors

print("=== Final Probe: A1+A2+B2 vs Baseline (3 seeds) ===")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 50
SEEDS = [42, 123, 456]

print("\n[1] Loading BRCA (cached after first seed)...")
ad_brca = load_dataset("brca_wu2021")


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
    return {
        "r_abs": pearson(p_last, x_last),
        "r_delta": pearson(p_last - x_prev, x_last - x_prev),
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

CHOSEN = dict(
    stalk_dim=4,
    use_orthogonal_maps=True,    # A1
    use_implicit_diffusion=True, # A2 (helps)
    predict_delta=False,         # A3 disabled (hurts in full stack with B1)
    use_nodal_film=False,        # B1 disabled (hurts)
    use_neural_ode=True,         # B2 (neutral)
    ode_hidden_dim=256, ode_solver="rk4",
)
# Residual variant: A3 re-enabled. Hypothesis: A3 hurt only in interaction
# with B1 (nodal FiLM amplified delta into noise). With B1 off, residual
# X(t)+delta restores absolute-scale signal lost in d=4 stalk compression.
CHOSEN_RESIDUAL = dict(
    stalk_dim=4,
    use_orthogonal_maps=True,
    use_implicit_diffusion=True,
    predict_delta=True,          # residual skip X(t)+delta
    use_nodal_film=False,
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


def run_one(name, seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    b_brca = adata_to_tsgnn_tensors(
        ad_brca, K=10, n_genes=500, k_neighbors=8, seed=seed, pseudotime_method="dpt"
    )
    X = b_brca["X_seq"].to(device)
    ei = b_brca["edge_index"].to(device)
    K, N, _ = X.shape
    esm = torch.randn(1280, device=device) * 0.1

    base = dict(
        num_nodes=N, num_edges=ei.shape[1],
        input_dim=1, esm_dim=1280, conditioning_dim=128,
        edge_index=ei, num_diffusion_steps=3,
        use_allele_conditioning=True,
        use_vectorized=True,
        use_normalized_laplacian=True,
    )
    base.update(cfg)
    torch.manual_seed(seed)
    model = TSGNN(**base).to(device)
    loss = train_model(model, X, esm)
    res = eval_model(model, X, esm)
    print(f"  [{name} seed={seed}]  loss={loss:.4f}  r_abs={res['r_abs']:.4f}  r_delta={res['r_delta']:.4f}")
    return res


print("\n[2] Running chosen (A1+A2+B2, no residual) over 3 seeds...")
chosen_rd = [run_one("chosen", s, CHOSEN) for s in SEEDS]

print("\n[3] Running chosen+residual (A1+A2+B2, predict_delta=True) over 3 seeds...")
residual_rd = [run_one("chosen+resid", s, CHOSEN_RESIDUAL) for s in SEEDS]

print("\n[4] Running baseline over 3 seeds...")
baseline_rd = [run_one("baseline", s, BASELINE) for s in SEEDS]

print("\n=== FINAL RESULTS (3 seeds) ===")
print(f"{'Seed':<8} {'chosen r_a':>12} {'chosen r_d':>12} {'resid r_a':>12} {'resid r_d':>12} {'base r_a':>12} {'base r_d':>12}")
print("-" * 90)
for i, s in enumerate(SEEDS):
    print(f"  {s:<6} {chosen_rd[i]['r_abs']:>12.4f} {chosen_rd[i]['r_delta']:>12.4f} "
          f"{residual_rd[i]['r_abs']:>12.4f} {residual_rd[i]['r_delta']:>12.4f} "
          f"{baseline_rd[i]['r_abs']:>12.4f} {baseline_rd[i]['r_delta']:>12.4f}")

def stats(name, results):
    abs_vals = [r["r_abs"] for r in results]
    delta_vals = [r["r_delta"] for r in results]
    print(f"  {name:<14}  r_abs: {np.mean(abs_vals):+.4f}±{np.std(abs_vals):.4f}   "
          f"r_delta: {np.mean(delta_vals):+.4f}±{np.std(delta_vals):.4f}   "
          f"P1pass: {sum(r > 0.35 for r in delta_vals)}/3")

print()
stats("chosen", chosen_rd)
stats("chosen+resid", residual_rd)
stats("baseline", baseline_rd)

# Paired comparison: residual vs chosen
resid_vs_chosen = [residual_rd[i]["r_delta"] - chosen_rd[i]["r_delta"] for i in range(len(SEEDS))]
resid_vs_chosen_abs = [residual_rd[i]["r_abs"] - chosen_rd[i]["r_abs"] for i in range(len(SEEDS))]
chosen_vs_base = [chosen_rd[i]["r_delta"] - baseline_rd[i]["r_delta"] for i in range(len(SEEDS))]
resid_vs_base = [residual_rd[i]["r_delta"] - baseline_rd[i]["r_delta"] for i in range(len(SEEDS))]

print(f"\n[paired] residual vs chosen:  Δr_abs={np.mean(resid_vs_chosen_abs):+.4f}  Δr_delta={np.mean(resid_vs_chosen):+.4f}  ({sum(d > 0 for d in resid_vs_chosen)}/3 wins resid)")
print(f"[paired] chosen   vs base:    Δr_delta={np.mean(chosen_vs_base):+.4f}  ({sum(d > 0 for d in chosen_vs_base)}/3 wins chosen)")
print(f"[paired] residual vs base:    Δr_delta={np.mean(resid_vs_base):+.4f}  ({sum(d > 0 for d in resid_vs_base)}/3 wins resid)")

print(f"\nVerdict (residual hypothesis):")
if np.mean(resid_vs_chosen_abs) > 0.20 and np.mean(resid_vs_chosen) >= -0.02:
    print("  RESIDUAL fixes r_abs without hurting r_delta. SHIP residual variant.")
elif np.mean(resid_vs_chosen_abs) > 0.10:
    print("  RESIDUAL helps r_abs partially. Consider for cross-dataset.")
else:
    print("  RESIDUAL hypothesis rejected. Keep chosen (no residual).")
