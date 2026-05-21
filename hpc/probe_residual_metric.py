"""
FASE 0 — Residualized Metric Probe (r_Δ⊥)

Tests whether model captures sample-specific signal BEYOND mean drift direction.

Method:
  m = Δ̄_train / ‖Δ̄_train‖   (mean drift direction from training set)
  P⊥ = I - mm^T               (projector orthogonal to mean drift)
  r_Δ⊥ = corr(P⊥ Δ̂, P⊥ Δ_true)

Decision:
  Δr_Δ⊥(model - mean) > 0.10 → Path A (full TSGNN-v2)
  Δr_Δ⊥(model - mean) ∈ [0.03, 0.10] → Path C (minimal fix)
  Δr_Δ⊥(model - mean) < 0.03 → Path B (reframe)
  r_Δ⊥(model) ≤ 0 → Path D (pivot)

Baselines compared:
  - chosen model (current arch)
  - mean predictor
  - identity
  - random predictor
  - X(t) regression to mean (predict X̄)
"""
import sys, os
sys.path.insert(0, "/home/3393519/ts-gnn/src")
os.chdir("/home/3393519/ts-gnn")

import torch
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

from tsgnn.data.registry import load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors, align_to_gene_set
from tsgnn.model.tsgnn import TSGNN

print("=== FASE 0: Residualized Metric Probe (r_Δ⊥) ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N_GENES, K_BINS, KNN, N_EPOCHS = 500, 10, 8, 50

CHOSEN = dict(
    stalk_dim=4, use_orthogonal_maps=True, use_implicit_diffusion=True,
    predict_delta=False, use_nodal_film=False, use_neural_ode=True,
    ode_hidden_dim=256, ode_solver="rk4",
)

def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def residualized_corr(delta_pred, delta_true, m):
    """Project deltas onto subspace orthogonal to m, compute correlation."""
    delta_pred = np.asarray(delta_pred).ravel()
    delta_true = np.asarray(delta_true).ravel()
    m = np.asarray(m).ravel()
    m = m / (np.linalg.norm(m) + 1e-12)
    # P⊥ x = x - (m·x) m
    proj_pred = delta_pred - np.dot(m, delta_pred) * m
    proj_true = delta_true - np.dot(m, delta_true) * m
    return pearson(proj_pred, proj_true)

# ─────────────────────────────────────────────────────────────────
# Load + prep (same as bias probe v2)
# ─────────────────────────────────────────────────────────────────

print("\n[load] BRCA + CRC...")
ad_brca = load_dataset("brca_wu2021"); ad_brca.var_names_make_unique()
ad_crc = load_dataset("crc_pelka2021"); ad_crc.var_names_make_unique()

def gene_var_logcpm(adata):
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True).clip(min=1.0)
    X = np.log1p(X / lib * 1e4)
    return dict(zip(list(adata.var_names.astype(str)), X.var(axis=0)))

v_brca = gene_var_logcpm(ad_brca)
v_crc = gene_var_logcpm(ad_crc)
common = set(v_brca) & set(v_crc)
combined = {g: min(v_brca[g], v_crc[g]) for g in common}
top_combined = sorted(combined, key=combined.get, reverse=True)[:N_GENES]

ad_b = ad_brca[:, [g for g in top_combined if g in ad_brca.var_names]].copy()
ad_c = ad_crc[:, [g for g in top_combined if g in ad_crc.var_names]].copy()

b_b = adata_to_tsgnn_tensors(ad_b, K=K_BINS, n_genes=N_GENES, k_neighbors=KNN, seed=42, pseudotime_method="dpt")
b_c = adata_to_tsgnn_tensors(ad_c, K=K_BINS, n_genes=N_GENES, k_neighbors=KNN, seed=42, pseudotime_method="dpt")

inter = sorted(set(b_b["gene_names"]) & set(b_c["gene_names"]))
b_b = align_to_gene_set(b_b, inter)
b_c = align_to_gene_set(b_c, inter)

X_b_full = b_b["X_seq"].squeeze(-1).numpy()  # (K, N)
X_c_full = b_c["X_seq"].squeeze(-1).numpy()  # (K, N)
X_b = b_b["X_seq"].to(device)
ei_b = b_b["edge_index"].to(device)
X_c = b_c["X_seq"].to(device)
ei_c = b_c["edge_index"].to(device)

# ─────────────────────────────────────────────────────────────────
# Compute mean drift direction from BRCA training data
# ─────────────────────────────────────────────────────────────────

print("\n[mean drift] Computing m from BRCA training transitions")
# Δ̄_train = average of (X(t+1) - X(t)) across training timesteps
delta_train_brca = X_b_full[1:] - X_b_full[:-1]   # (K-1, N)
m_brca = delta_train_brca.mean(axis=0)             # (N,)
m_brca_norm = np.linalg.norm(m_brca)
print(f"  BRCA: m direction norm = {m_brca_norm:.6f}")
print(f"  BRCA: mean drift mean = {m_brca.mean():.6f}, std = {m_brca.std():.6f}")

delta_train_crc = X_c_full[1:] - X_c_full[:-1]
m_crc = delta_train_crc.mean(axis=0)
print(f"  CRC: m direction norm = {np.linalg.norm(m_crc):.6f}")

# ─────────────────────────────────────────────────────────────────
# Train chosen model on BRCA
# ─────────────────────────────────────────────────────────────────

print("\n[train] Chosen model on BRCA (50 epochs)...")
torch.manual_seed(42)
np.random.seed(42)
base = dict(num_nodes=X_b.shape[1], num_edges=ei_b.shape[1], input_dim=1, esm_dim=1280,
            conditioning_dim=128, edge_index=ei_b, num_diffusion_steps=3,
            use_allele_conditioning=True, use_vectorized=True, use_normalized_laplacian=True)
base.update(CHOSEN)
model = TSGNN(**base).to(device)

esm_fake = torch.randn(1280, device=device) * 0.1
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
val_k = max(1, K_BINS // 5)
X_train = X_b[:-val_k - 1]
Y_train = X_b[1:-val_k]
model.train()
for ep in range(N_EPOCHS):
    opt.zero_grad()
    out = model(X_train, esm_fake)
    preds = out[0] if isinstance(out, tuple) else out
    if isinstance(preds, list):
        preds = torch.stack(preds, dim=0)
    loss = ((preds - Y_train) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print(f"  Final loss: {loss.item():.6f}")

# ─────────────────────────────────────────────────────────────────
# Evaluate predictors on CRC last timestep
# ─────────────────────────────────────────────────────────────────

print("\n[eval] CRC last timestep predictions vs ground truth")

x_prev = X_c_full[-2]    # (N,)
x_true = X_c_full[-1]    # (N,)
delta_true = x_true - x_prev

# 1. Chosen model
model.eval()
with torch.no_grad():
    out = model(X_c[:-1], esm_fake)
    preds = out[0] if isinstance(out, tuple) else out
    if isinstance(preds, list):
        preds = torch.stack(preds, dim=0)
pred_model = preds[-1].cpu().numpy().squeeze()
delta_model = pred_model - x_prev

# 2. Mean predictor: predict X̄ over all timesteps
x_mean = X_c_full.mean(axis=0)
delta_mean = x_mean - x_prev

# 3. Identity: predict X(t-1) = X(t-2)
pred_identity = X_c_full[-2]
delta_identity = pred_identity - x_prev   # = 0

# 4. Identity-shifted: predict X(t-1) = previous transition (X[-2] - X[-3]) added
delta_identity_shift = X_c_full[-2] - X_c_full[-3]   # last observed delta

# 5. Random predictor (Gaussian noise)
np.random.seed(0)
delta_random = np.random.randn(*x_prev.shape) * delta_true.std()

# 6. Pure mean drift predictor: predict X(t) = X(t-1) + m_brca
delta_drift = m_brca

# ─────────────────────────────────────────────────────────────────
# Compute r_d (raw) and r_Δ⊥ (residualized)
# ─────────────────────────────────────────────────────────────────

predictors = {
    "chosen_model":      delta_model,
    "mean_predictor":    delta_mean,
    "identity":          delta_identity,
    "identity_shift":    delta_identity_shift,
    "random":            delta_random,
    "mean_drift_BRCA":   delta_drift,
}

# Use BRCA-derived m (cross-dataset transfer scenario)
m_use = m_brca

print(f"\nUsing mean drift direction m from: BRCA training")
print(f"\n{'Predictor':<20} {'r_d (raw)':>12} {'r_Δ⊥ (resid)':>14} {'Δ vs mean':>12}")
print("-" * 60)

results = {}
for name, dpred in predictors.items():
    r_d = pearson(dpred, delta_true)
    r_perp = residualized_corr(dpred, delta_true, m_use)
    results[name] = (r_d, r_perp)
    print(f"{name:<20} {r_d:>+12.4f} {r_perp:>+14.4f}")

# Compute deltas vs mean predictor
print(f"\nΔ vs mean_predictor (r_Δ⊥):")
mean_perp = results["mean_predictor"][1]
for name, (r_d, r_perp) in results.items():
    delta_perp = r_perp - mean_perp
    print(f"  {name:<20}  Δ = {delta_perp:>+.4f}")

# ─────────────────────────────────────────────────────────────────
# Decision logic
# ─────────────────────────────────────────────────────────────────

model_perp = results["chosen_model"][1]
mean_perp = results["mean_predictor"][1]
delta_signal = model_perp - mean_perp

print(f"\n{'='*60}")
print(f"DECISION GATE — r_Δ⊥(model) - r_Δ⊥(mean) = {delta_signal:+.4f}")
print(f"{'='*60}")

if model_perp <= 0:
    print(f"  Path D: PIVOT — model has no signal beyond mean drift")
    print(f"  Project should be abandoned or radically refocused")
elif delta_signal > 0.10:
    print(f"  Path A: PROCEED with TSGNN-v2 full rewrite")
    print(f"  Strong signal beyond mean drift, architecture investment justified")
elif delta_signal > 0.03:
    print(f"  Path C: MINIMAL FIX — so(d) + weighted sheaf, skip GRN/temporal")
    print(f"  Marginal signal, target 3-4/5 acceptance gates")
else:
    print(f"  Path B: REFRAME — paper to bioinformatics venue")
    print(f"  Model essentially captures only mean drift / regression-to-mean")

# Bonus: also evaluate on BRCA test split (in-domain)
print(f"\n[bonus] In-domain BRCA last-timestep")
x_prev_b = X_b_full[-2]
x_true_b = X_b_full[-1]
delta_true_b = x_true_b - x_prev_b

with torch.no_grad():
    out_b = model(X_b[:-1], esm_fake)
    preds_b = out_b[0] if isinstance(out_b, tuple) else out_b
    if isinstance(preds_b, list):
        preds_b = torch.stack(preds_b, dim=0)
pred_b = preds_b[-1].cpu().numpy().squeeze()
delta_b = pred_b - x_prev_b

# Use validation split for m (avoid leakage)
m_b_train = X_b_full[:-2].mean(axis=0) - X_b_full[:-3].mean(axis=0)  # rough mean drift from train portion
r_d_b = pearson(delta_b, delta_true_b)
r_perp_b = residualized_corr(delta_b, delta_true_b, m_b_train)
print(f"  BRCA in-domain: r_d = {r_d_b:+.4f}, r_Δ⊥ = {r_perp_b:+.4f}")

print("\n=== END FASE 0 PROBE ===")
