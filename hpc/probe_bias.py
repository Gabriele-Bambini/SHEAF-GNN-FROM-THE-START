"""
Bias / sanity probe — post-hoc validation of cross-dataset claim.

Tests:
  T1. Identity baseline: pearson(X[t-1]-X[t-2], X[t]-X[t-2]) — trivial autocorrelation r_d
  T2. Mean predictor: pearson(mean(X)-X[t-1], X[t]-X[t-1]) — constant prediction
  T3. Hard HVG bias: select top-500 by var_brca ONLY (no CRC pre-screen). Re-eval.
  T4. CUDA non-determinism: run same seed=42 three times, measure std.
  T5. FiLM dead test: predict CRC with esm=zeros vs esm=randn. Compare.
  T6. Pseudotime artifact: shuffle CRC bin order. r_d should crash if signal real.
  T7. Graph ablation: replace edge_index with random edges. r_d should drop if sheaf matters.

Decision: if T1/T2 already give > 0.60 r_d on CRC → "transfer" claim weak.
         If T5 diff < 0.01 → FiLM is dead, allele-specificity claim invalid.
         If T7 r_d ≈ 0.70 → graph structure unused, sheaf claim weak.
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

print("=== Bias / Sanity Probe — post-hoc validation ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
N_GENES = 500
K_BINS = 10
KNN = 8
N_EPOCHS = 50


def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def gene_variance(adata):
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True).clip(min=1.0)
    X = np.log1p(X / lib * 1e4)
    return dict(zip(list(adata.var_names.astype(str)), X.var(axis=0)))


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


CHOSEN = dict(
    stalk_dim=4, use_orthogonal_maps=True, use_implicit_diffusion=True,
    predict_delta=False, use_nodal_film=False, use_neural_ode=True,
    ode_hidden_dim=256, ode_solver="rk4",
)

print("\n[load] BRCA + CRC...")
ad_brca = load_dataset("brca_wu2021"); ad_brca.var_names_make_unique()
ad_crc = load_dataset("crc_pelka2021"); ad_crc.var_names_make_unique()

# ===== Combined-HVG selection (matches probe_xdataset v2) =====
v_brca = gene_variance(ad_brca)
v_crc = gene_variance(ad_crc)
common = set(v_brca) & set(v_crc)
combined = {g: min(v_brca[g], v_crc[g]) for g in common}
top_combined = sorted(combined, key=combined.get, reverse=True)[:N_GENES]

# ===== T3: BRCA-only HVG (no CRC pre-screen) =====
top_brca_only = sorted(v_brca, key=v_brca.get, reverse=True)[:N_GENES]


def prep(top_genes, seed=42):
    ad_b = ad_brca[:, [g for g in top_genes if g in ad_brca.var_names]].copy()
    ad_c = ad_crc[:, [g for g in top_genes if g in ad_crc.var_names]].copy()
    b_b = adata_to_tsgnn_tensors(ad_b, K=K_BINS, n_genes=N_GENES, k_neighbors=KNN,
                                  seed=seed, pseudotime_method="dpt")
    b_c = adata_to_tsgnn_tensors(ad_c, K=K_BINS, n_genes=N_GENES, k_neighbors=KNN,
                                  seed=seed, pseudotime_method="dpt")
    inter = sorted(set(b_b["gene_names"]) & set(b_c["gene_names"]))
    b_b = align_to_gene_set(b_b, inter)
    b_c = align_to_gene_set(b_c, inter)
    return b_b, b_c, len(inter)


def make_model_train(b_brca, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    X = b_brca["X_seq"].to(device)
    ei = b_brca["edge_index"].to(device)
    K, N, _ = X.shape
    esm = torch.randn(1280, device=device) * 0.1
    base = dict(num_nodes=N, num_edges=ei.shape[1], input_dim=1, esm_dim=1280,
                conditioning_dim=128, edge_index=ei, num_diffusion_steps=3,
                use_allele_conditioning=True, use_vectorized=True,
                use_normalized_laplacian=True)
    base.update(CHOSEN)
    torch.manual_seed(seed)
    model = TSGNN(**base).to(device)
    train_model(model, X, esm)
    return model, X, ei, esm


# === T1: Identity baseline (predict X(t)) on CRC ===
print("\n[T1] Identity baseline r_d on CRC (predict X[t]=X[t-1])")
b_brca, b_crc, n1 = prep(top_combined)
X_crc = b_crc["X_seq"].squeeze(-1).numpy()  # (K, N)
# predict: X̂[t]=X[t-1] → r_d = pearson(X[t-1]-X[t-2], X[t]-X[t-2])
identity_rd = pearson(X_crc[-2] - X_crc[-3], X_crc[-1] - X_crc[-3])
print(f"  CRC identity r_d = {identity_rd:+.4f}")

# === T2: Mean predictor on CRC ===
print("\n[T2] Mean predictor r_d on CRC (predict bin-mean)")
mean_pred = X_crc.mean(axis=0)
mean_rd = pearson(mean_pred - X_crc[-2], X_crc[-1] - X_crc[-2])
print(f"  CRC mean-predictor r_d = {mean_rd:+.4f}")

# === T4: CUDA non-det (3 runs same seed) ===
print("\n[T4] CUDA non-determinism (seed=42 × 3)")
ndet_rd = []
for trial in range(3):
    model, X_b, ei_b, esm_b = make_model_train(b_brca, seed=42)
    X_c = b_crc["X_seq"].to(device)
    ei_c = b_crc["edge_index"].to(device)
    r = eval_model(model, X_c, esm_b, edge_index=ei_c)
    ndet_rd.append(r["r_delta"])
    print(f"  trial {trial+1}: CRC r_d = {r['r_delta']:+.4f}")
print(f"  std(ndet) = {np.std(ndet_rd):.4f}  (compare to seed-std 0.004 from job 483052)")

# Train ONCE for T5/T6/T7 (all use same trained chosen model)
model, X_b, ei_b, esm_train = make_model_train(b_brca, seed=42)
X_c = b_crc["X_seq"].to(device)
ei_c = b_crc["edge_index"].to(device)

# === T5: FiLM dead test ===
print("\n[T5] FiLM dead test (zero ESM vs randn ESM)")
torch.manual_seed(0)
esm_zero = torch.zeros(1280, device=device)
esm_rand1 = torch.randn(1280, device=device) * 0.1
esm_rand2 = torch.randn(1280, device=device) * 0.1
r_zero = eval_model(model, X_c, esm_zero, edge_index=ei_c)
r_r1 = eval_model(model, X_c, esm_rand1, edge_index=ei_c)
r_r2 = eval_model(model, X_c, esm_rand2, edge_index=ei_c)
print(f"  zero ESM:  r_d = {r_zero['r_delta']:+.4f}")
print(f"  randn 1:   r_d = {r_r1['r_delta']:+.4f}")
print(f"  randn 2:   r_d = {r_r2['r_delta']:+.4f}")
film_swing = max(r_zero['r_delta'], r_r1['r_delta'], r_r2['r_delta']) - min(r_zero['r_delta'], r_r1['r_delta'], r_r2['r_delta'])
print(f"  FiLM swing (max-min) = {film_swing:+.4f}  → if <0.01, FiLM is dead")

# === T6: Pseudotime shuffle ===
print("\n[T6] Pseudotime shuffle on CRC (bin order randomized)")
torch.manual_seed(0)
perm = torch.randperm(K_BINS)
X_c_shuf = X_c[perm]
r_shuf = eval_model(model, X_c_shuf, esm_train, edge_index=ei_c)
print(f"  shuffled r_d = {r_shuf['r_delta']:+.4f}  (original was 0.70)")

# === T7: Graph ablation ===
print("\n[T7] Graph ablation on CRC (random edge_index)")
torch.manual_seed(0)
N_crc = X_c.shape[1]
ei_random = torch.randint(0, N_crc, (2, ei_c.shape[1]), device=device)
r_rand_g = eval_model(model, X_c, esm_train, edge_index=ei_random)
print(f"  random graph r_d = {r_rand_g['r_delta']:+.4f}")

# === T3: BRCA-only HVG ===
print("\n[T3] HVG selection bias — top-500 by var_brca ONLY")
b_brca3, b_crc3, n3 = prep(top_brca_only)
print(f"  n_common after BRCA-only HVG = {n3}")
if n3 >= 50:
    model3, X_b3, ei_b3, esm3 = make_model_train(b_brca3, seed=42)
    X_c3 = b_crc3["X_seq"].to(device)
    ei_c3 = b_crc3["edge_index"].to(device)
    r3 = eval_model(model3, X_c3, esm3, edge_index=ei_c3)
    print(f"  CRC r_d (BRCA-only HVG) = {r3['r_delta']:+.4f}  (combined-HVG was 0.70)")
else:
    print(f"  too few common genes; skipping eval")

# ===== SUMMARY =====
print("\n=== BIAS SUMMARY ===")
print(f"  T1 identity        CRC r_d = {identity_rd:+.4f}    (ref: chosen 0.70)")
print(f"  T2 mean predictor  CRC r_d = {mean_rd:+.4f}")
print(f"  T4 cuda non-det    std     = {np.std(ndet_rd):.4f}    (ref: seed-std 0.004)")
print(f"  T5 FiLM swing                = {film_swing:+.4f}    (>0.01 = active, <0.01 = dead)")
print(f"  T6 shuffle         CRC r_d = {r_shuf['r_delta']:+.4f}    (should crash)")
print(f"  T7 random graph    CRC r_d = {r_rand_g['r_delta']:+.4f}    (should drop)")
print()
print("VERDICTS:")
print(f"  identity-vs-chosen    Δ = {0.70 - identity_rd:+.4f}    {'PASS' if (0.70 - identity_rd) > 0.05 else 'FAIL'}")
print(f"  FiLM active           {'PASS' if film_swing > 0.01 else 'FAIL — allele claim invalid'}")
print(f"  pseudotime real       {'PASS' if r_shuf['r_delta'] < 0.30 else 'FAIL — bin-order artifact'}")
print(f"  graph used            {'PASS' if (0.70 - r_rand_g['r_delta']) > 0.05 else 'FAIL — sheaf claim weak'}")
print(f"  cuda-vs-seed std      {'OK' if np.std(ndet_rd) < 0.004 * 2 else 'WARN'}  (cuda noise dominates if comparable)")
