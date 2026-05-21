"""
Diagnostic probe — investigate root causes of T5/T6/T7 failures.

T5 FAILURE: FiLM swing 0.001 (dead).
  → Check: Are FiLM γ/β actually different for zero vs random ESM?
  → Check: Gradient flow reaching FiLM during training?
  → Check: Weight norms at inference?

T6 FAILURE: Shuffled bins work better (artifact).
  → Check: Is temporal adjacency used in graph?
  → Check: Shuffle graph structure too. Does that hurt more than shuffle bins?

T7 FAILURE: Random edges ≈ real edges (unused).
  → Check: What is the graph learning? Node aggregation?
  → Check: Attention weights on edges. Are they uniform?
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

print("=== Diagnostic Probe — T5/T6/T7 Root Cause Analysis ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Same config as bias probe
N_GENES = 500
K_BINS = 10
KNN = 8
N_EPOCHS = 50

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

# Load + prep
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
        loss = ((preds - Y_train) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.item()

# ─────────────────────────────────────────────────────────────────────────────
# T5 DIAGNOSIS: FiLM dead
# ─────────────────────────────────────────────────────────────────────────────

print("\n[T5-DIAG] FiLM Conditioning Analysis")

torch.manual_seed(42)
np.random.seed(42)
base = dict(num_nodes=X_b.shape[1], num_edges=ei_b.shape[1], input_dim=1, esm_dim=1280,
            conditioning_dim=128, edge_index=ei_b, num_diffusion_steps=3,
            use_allele_conditioning=True, use_vectorized=True,
            use_normalized_laplacian=True)
base.update(CHOSEN)
model = TSGNN(**base).to(device)

# Train
esm_fake = torch.randn(1280, device=device) * 0.1
train_model(model, X_b, esm_fake, n_epochs=N_EPOCHS)

# Inference hooks: capture FiLM outputs
film_gamma_list = []
film_beta_list = []
original_film_forward = model.allele_film.forward

def hooked_forward(esm_emb, gru_out, edge_index=None):
    global film_gamma_list, film_beta_list
    if esm_emb.dim() == 1:
        esm_emb = esm_emb.unsqueeze(0)
    z = model.allele_film.allele_encoder(esm_emb)
    film_params = model.allele_film.film_generator(z)
    gamma, beta = film_params.chunk(2, dim=-1)
    gamma = gamma + 1.0
    film_gamma_list.append(gamma.detach().cpu().numpy())
    film_beta_list.append(beta.detach().cpu().numpy())
    modulated = gamma * gru_out + beta
    return modulated

model.allele_film.forward = hooked_forward

# Test 1: zero ESM
film_gamma_list = []
film_beta_list = []
esm_zero = torch.zeros(1280, device=device)
model.eval()
with torch.no_grad():
    _ = model(X_c[:-1], esm_zero)
gamma_zero = np.concatenate(film_gamma_list, axis=0)  # (K*E, modulation_dim)
beta_zero = np.concatenate(film_beta_list, axis=0)

# Test 2: random ESM
film_gamma_list = []
film_beta_list = []
torch.manual_seed(0)
esm_rand = torch.randn(1280, device=device) * 0.1
with torch.no_grad():
    _ = model(X_c[:-1], esm_rand)
gamma_rand = np.concatenate(film_gamma_list, axis=0)
beta_rand = np.concatenate(film_beta_list, axis=0)

print(f"  γ (zero ESM)  : mean={gamma_zero.mean():.6f}, std={gamma_zero.std():.6f}")
print(f"  γ (rand ESM)  : mean={gamma_rand.mean():.6f}, std={gamma_rand.std():.6f}")
print(f"  Δγ (L2 dist)  : {np.linalg.norm(gamma_zero - gamma_rand):.6f}")
print(f"  β (zero ESM)  : mean={beta_zero.mean():.6f}, std={beta_zero.std():.6f}")
print(f"  β (rand ESM)  : mean={beta_rand.mean():.6f}, std={beta_rand.std():.6f}")
print(f"  Δβ (L2 dist)  : {np.linalg.norm(beta_zero - beta_rand):.6f}")

if np.linalg.norm(gamma_zero - gamma_rand) < 1e-4 and np.linalg.norm(beta_zero - beta_rand) < 1e-4:
    print(f"  ⚠ VERDICT: FiLM NOT RESPONSIVE to ESM variation. Weights may not have trained.")
else:
    print(f"  ✓ FiLM responsive (but r_d still same — may be masked by other signal).")

# Restore
model.allele_film.forward = original_film_forward

# ─────────────────────────────────────────────────────────────────────────────
# T7 DIAGNOSIS: Graph unused
# ─────────────────────────────────────────────────────────────────────────────

print("\n[T7-DIAG] Graph Structure Importance")

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

esm_train = torch.randn(1280, device=device) * 0.1

# Original graph
r_orig = eval_model(model, X_c, esm_train, edge_index=ei_c)
print(f"  Original KNN graph r_d = {r_orig['r_delta']:+.4f}")

# Random graph (same cardinality)
torch.manual_seed(0)
N_crc = X_c.shape[1]
ei_random = torch.randint(0, N_crc, (2, ei_c.shape[1]), device=device)
r_rand = eval_model(model, X_c, esm_train, edge_index=ei_random)
print(f"  Random graph r_d       = {r_rand['r_delta']:+.4f}")

# Fully connected (all-to-all)
ei_full = torch.stack([
    torch.repeat_interleave(torch.arange(N_crc, device=device), N_crc),
    torch.tile(torch.arange(N_crc, device=device), (N_crc,))
])
r_full = eval_model(model, X_c, esm_train, edge_index=ei_full)
print(f"  Fully connected r_d    = {r_full['r_delta']:+.4f}")

# No edges (identity aggregation)
ei_diag = torch.stack([torch.arange(N_crc, device=device), torch.arange(N_crc, device=device)])
r_diag = eval_model(model, X_c, esm_train, edge_index=ei_diag)
print(f"  Diagonal (no edges) r_d = {r_diag['r_delta']:+.4f}")

deltas = [
    ("orig → random", r_orig['r_delta'] - r_rand['r_delta']),
    ("orig → full", r_orig['r_delta'] - r_full['r_delta']),
    ("orig → diag", r_orig['r_delta'] - r_diag['r_delta']),
]
for label, delta in deltas:
    print(f"  Δ {label:15s} = {delta:+.4f}")

if all(abs(d) < 0.02 for _, d in deltas):
    print(f"  ⚠ VERDICT: Graph structure has MINIMAL impact. Model ignores edge structure.")
else:
    print(f"  ✓ Graph matters (structure learned).")

# ─────────────────────────────────────────────────────────────────────────────
# T6 DIAGNOSIS: Pseudotime artifact
# ─────────────────────────────────────────────────────────────────────────────

print("\n[T6-DIAG] Pseudotime and Temporal Adjacency")

# Original order
r_orig_seq = eval_model(model, X_c, esm_train, edge_index=ei_c)
print(f"  Original bin order r_d = {r_orig_seq['r_delta']:+.4f}")

# Shuffle bin order
torch.manual_seed(0)
perm = torch.randperm(K_BINS)
X_c_shuf = X_c[perm]
r_shuf_seq = eval_model(model, X_c_shuf, esm_train, edge_index=ei_c)
print(f"  Shuffled bin order r_d = {r_shuf_seq['r_delta']:+.4f}")

# Shuffle BOTH bin order AND graph (disconnect temporal adjacency)
# Temporal adjacency in default KNN: edges mostly within-timestep or nearest-neighbor timesteps
# Disrupting both removes temporal structure entirely
ei_shuf_graph = ei_c[:, torch.randperm(ei_c.shape[1])]
r_shuf_both = eval_model(model, X_c_shuf, esm_train, edge_index=ei_shuf_graph)
print(f"  Shuf bins + shuf graph r_d = {r_shuf_both['r_delta']:+.4f}")

print(f"  Δ (shuf − orig) = {r_shuf_seq['r_delta'] - r_orig_seq['r_delta']:+.4f}")
print(f"  Δ (shuf_both − orig) = {r_shuf_both['r_delta'] - r_orig_seq['r_delta']:+.4f}")

if r_shuf_seq['r_delta'] > r_orig_seq['r_delta']:
    print(f"  ⚠ VERDICT: Shuffling bins IMPROVES r_d. Model learns cell distribution, not temporal sequence.")
    print(f"           Temporal adjacency in graph may be HARMFUL (overfitting to bin order).")
else:
    print(f"  ✓ Temporal structure matters.")

print("\n=== END DIAGNOSTIC PROBE ===")
