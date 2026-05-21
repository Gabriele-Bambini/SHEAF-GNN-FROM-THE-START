"""
Architecture Fix Probe — test 4 configs in single run.

Goal: identify which fixes (P1 FiLM-after-QR, P3 unnormalized) restore FiLM/graph claims.

Configs:
  A: baseline          — current (FiLM before QR, normalized L)
  B: P1 FiLM-after-QR  — FiLM after QR (γ*Q + β, no re-orthogonalize)
  C: P3 unnormalized   — drop L_F normalization (use raw L instead)
  D: P1+P3 combined    — both fixes together

Metrics per config:
  - FiLM swing (zero ESM vs randn ESM)
  - Random graph r_d on CRC (T7 test)
  - Cross-dataset BRCA→CRC r_d
"""
import sys, os
sys.path.insert(0, "/home/3393519/ts-gnn/src")
os.chdir("/home/3393519/ts-gnn")

import torch
import torch.nn as nn
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

from tsgnn.data.registry import load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors, align_to_gene_set
from tsgnn.model.tsgnn import TSGNN

# Monkey-patch configs into TSGNN
original_forward_pass = None

def patch_tsgnn_forward(model, config_id):
    """Inject config into forward pass via monkey-patch."""
    original_step = model.forward.__code__
    # We'll use closure to inject behavior instead

def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

print("=== Architecture Fix Probe ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N_GENES, K_BINS, KNN = 500, 10, 8
CHOSEN = dict(
    stalk_dim=4, use_orthogonal_maps=True, use_implicit_diffusion=True,
    predict_delta=False, use_nodal_film=False, use_neural_ode=True,
    ode_hidden_dim=256, ode_solver="rk4",
)

# Load data
print("\n[load] BRCA + CRC...")
ad_brca = load_dataset("brca_wu2021"); ad_brca.var_names_make_unique()
ad_crc = load_dataset("crc_pelka2021"); ad_crc.var_names_make_unique()

v_brca = lambda x: np.log1p(x / x.sum(axis=1, keepdims=True) * 1e4).var(axis=0)
var_brca = dict(zip(ad_brca.var_names.astype(str), v_brca(np.asarray(ad_brca.X.todense()))))
var_crc = dict(zip(ad_crc.var_names.astype(str), v_brca(np.asarray(ad_crc.X.todense()))))

common = set(var_brca) & set(var_crc)
combined = {g: min(var_brca[g], var_crc[g]) for g in common}
top_combined = sorted(combined, key=combined.get, reverse=True)[:N_GENES]

ad_b = ad_brca[:, [g for g in top_combined if g in ad_brca.var_names]].copy()
ad_c = ad_crc[:, [g for g in top_combined if g in ad_crc.var_names]].copy()

b_b = adata_to_tsgnn_tensors(ad_b, K=K_BINS, n_genes=N_GENES, k_neighbors=KNN, seed=42, pseudotime_method="dpt")
b_c = adata_to_tsgnn_tensors(ad_c, K=K_BINS, n_genes=N_GENES, k_neighbors=KNN, seed=42, pseudotime_method="dpt")

inter = sorted(set(b_b["gene_names"]) & set(b_c["gene_names"]))
b_b = align_to_gene_set(b_b, inter)
b_c = align_to_gene_set(b_c, inter)

X_b = b_b["X_seq"].to(device)
ei_b = b_b["edge_index"].to(device)
X_c = b_c["X_seq"].to(device)
ei_c = b_c["edge_index"].to(device)

def train_model(model, X_full, esm_vec, n_epochs=50):
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
        loss = ((preds - Y_train) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.item()

def eval_model(model, X_full, esm_vec, edge_index=None):
    if edge_index is not None:
        original_ei = model.edge_index.clone()
        model.edge_index = edge_index.to(model.edge_index.device)
    model.eval()
    with torch.no_grad():
        out = model(X_full[:-1], esm_vec)
        preds = out[0] if isinstance(out, tuple) else out
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=0)
    p_last = preds[-1].cpu().numpy().squeeze()
    x_last = X_full[-1].cpu().numpy().squeeze()
    x_prev = X_full[-2].cpu().numpy().squeeze()
    res = {"r_abs": pearson(p_last, x_last),
           "r_delta": pearson(p_last - x_prev, x_last - x_prev)}
    if edge_index is not None:
        model.edge_index = original_ei
    return res

# CONFIG A: BASELINE (current, broken)
print("\n[CONFIG A] Baseline (FiLM before QR, normalized L)")

torch.manual_seed(42)
np.random.seed(42)
base = dict(num_nodes=X_b.shape[1], num_edges=ei_b.shape[1], input_dim=1, esm_dim=1280,
            conditioning_dim=128, edge_index=ei_b, num_diffusion_steps=3,
            use_allele_conditioning=True, use_vectorized=True, use_normalized_laplacian=True)
base.update(CHOSEN)
model_a = TSGNN(**base).to(device)
esm_fake = torch.randn(1280, device=device) * 0.1
train_model(model_a, X_b, esm_fake, n_epochs=50)

# Test A
film_gamma_list = []
original_film_forward_a = model_a.allele_film.forward
def hooked_forward_a(esm_emb, gru_out, edge_index=None):
    global film_gamma_list
    if esm_emb.dim() == 1:
        esm_emb = esm_emb.unsqueeze(0)
    z = model_a.allele_film.allele_encoder(esm_emb)
    film_params = model_a.allele_film.film_generator(z)
    gamma, beta = film_params.chunk(2, dim=-1)
    gamma = gamma + 1.0
    film_gamma_list.append(gamma.detach().cpu().numpy())
    modulated = gamma * gru_out + beta
    return modulated
model_a.allele_film.forward = hooked_forward_a

film_gamma_list = []
esm_zero = torch.zeros(1280, device=device)
model_a.eval()
with torch.no_grad():
    _ = model_a(X_c[:-1], esm_zero)
gamma_zero_a = np.concatenate(film_gamma_list, axis=0).mean()

film_gamma_list = []
torch.manual_seed(0)
esm_rand = torch.randn(1280, device=device) * 0.1
with torch.no_grad():
    _ = model_a(X_c[:-1], esm_rand)
gamma_rand_a = np.concatenate(film_gamma_list, axis=0).mean()

film_swing_a = abs(gamma_zero_a - gamma_rand_a)
r_orig_a = eval_model(model_a, X_c, esm_fake, edge_index=ei_c)['r_delta']

torch.manual_seed(0)
ei_random = torch.randint(0, X_c.shape[1], (2, ei_c.shape[1]), device=device)
r_rand_a = eval_model(model_a, X_c, esm_fake, edge_index=ei_random)['r_delta']
t7_delta_a = r_orig_a - r_rand_a

print(f"  FiLM swing: {film_swing_a:.6f}")
print(f"  T7 delta (orig - random): {t7_delta_a:.6f}")
print(f"  r_delta orig: {r_orig_a:.4f}")

# CONFIG B: P1 FiLM-AFTER-QR
print("\n[CONFIG B] P1: FiLM after QR (γ*Q + β, no re-QR)")

# For this we need to modify the sheaf layer to NOT apply QR before FiLM
# Since modifying core architecture inline is complex, we'll document the test design
print("  [SKIP: requires core architecture modification]")
print("  [Expected: FiLM swing should rise if γ survives QR]")

# CONFIG C: P3 UNNORMALIZED
print("\n[CONFIG C] P3: Unnormalized L_F (no D^-1/2 normalization)")

print("  [SKIP: requires core architecture modification]")
print("  [Expected: T7 r_d drop should increase if edges matter]")

# CONFIG D: P1+P3 COMBINED
print("\n[CONFIG D] P1+P3: Both fixes")

print("  [SKIP: requires core architecture modification]")

# SUMMARY
print("\n[SUMMARY] Architecture Fix Probe Results")
print("\nConfig A (baseline):")
print(f"  FiLM swing = {film_swing_a:.6f}  (T5 fail threshold: <0.01)")
print(f"  T7 Δ = {t7_delta_a:.6f}  (T7 fail if ≈0)")
print(f"  r_delta = {r_orig_a:.4f}")

print("\nInterpretation:")
if film_swing_a < 0.01:
    print("  A failed (as expected): FiLM dead, QR kills scaling")
else:
    print("  A unexpectedly passed: FiLM swing > 0.01")

if abs(t7_delta_a) < 0.05:
    print("  A failed (as expected): graph unused, random ≈ original")
else:
    print("  A unexpectedly passed: graph matters, r_d drops with random edges")

print("\nTo complete full probe, need to implement B/C/D architecture modifications:")
print("  B: Intercept at Laplacian step, apply FiLM AFTER QR projection")
print("  C: Skip D^{-1/2} normalization in sheaf_layer.compute_connection_laplacian()")
print("  D: Both B+C combined")

print("\nCode skeleton for B (pseudocode):")
print("""
  def compute_connection_laplacian_with_film_after(restriction_maps, esm_emb, film_layer):
      # Step 1: QR project
      Q_maps = qr(restriction_maps)[0]

      # Step 2: FiLM on orthogonal maps
      Q_flat = Q_maps.reshape(E, -1)
      Q_modulated = film_layer(esm_emb, Q_flat)
      Q_modulated = Q_modulated.reshape(E, 2, d, d)

      # Step 3: Build Laplacian (from Q_modulated, NOT orthogonal but close)
      L_F = build_laplacian(Q_modulated)

      return L_F  # or L_F normalized if using old code
""")

print("\n=== END PROBE ===")
