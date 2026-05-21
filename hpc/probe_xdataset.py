"""
Cross-dataset probe v2: BRCA → CRC zero-shot with proper gene selection.

v1 (job 483028) produced invalid results: top-500 by variance computed
SEPARATELY on each dataset → only 10 genes in intersection → overfitting on
80 data points → fake r_delta = 0.95.

v2 fix:
  1. Load BRCA + CRC AnnData
  2. Make var_names unique (BRCA has duplicates)
  3. Restrict to genes present in BOTH
  4. Compute log-CPM variance per gene in each dataset
  5. Rank by min(var_brca, var_crc) → robust combined ranking
  6. Take top-500 from this combined ranking
  7. Subset both AnnData to this fixed gene set
  8. Adapt both with n_genes=500 (now forced consistent)
  9. Train chosen on BRCA, eval zero-shot on CRC

Decision rule:
  CRC r_d > 0.30: GO
  CRC r_d in [0.20, 0.30]: PARTIAL
  CRC r_d < 0.20: NO-GO
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

print("=== Cross-Dataset Probe v2: BRCA → CRC zero-shot (combined HVG) ===")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 50
SEEDS = [42, 123, 456]
N_GENES = 500
K_BINS = 10
KNN = 8


def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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
    res = {
        "r_abs": pearson(p_last, x_last),
        "r_delta": pearson(p_last - x_prev, x_last - x_prev),
    }
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
    stalk_dim=4,
    use_orthogonal_maps=True,
    use_implicit_diffusion=True,
    predict_delta=False,
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


def gene_variance(adata):
    """log-CPM-like normalization, return dict gene_name -> variance."""
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True).clip(min=1.0)
    X = np.log1p(X / lib * 1e4)
    var = X.var(axis=0)
    return dict(zip(list(adata.var_names.astype(str)), var))


print("\n[1] Load BRCA + CRC AnnData...")
ad_brca = load_dataset("brca_wu2021")
ad_crc = load_dataset("crc_pelka2021")
print(f"  BRCA: {ad_brca.shape}")
print(f"  CRC:  {ad_crc.shape}")

# Make var_names unique
ad_brca.var_names_make_unique()
ad_crc.var_names_make_unique()

print("\n[2] Compute per-dataset log-CPM gene variance...")
v_brca = gene_variance(ad_brca)
v_crc = gene_variance(ad_crc)
print(f"  BRCA genes scored: {len(v_brca)}")
print(f"  CRC  genes scored: {len(v_crc)}")

print("\n[3] Compute combined variance ranking on intersection...")
common_genes = set(v_brca.keys()) & set(v_crc.keys())
print(f"  Genes present in both: {len(common_genes)}")
# Use min variance — gene must be informative in BOTH datasets
combined = {g: min(v_brca[g], v_crc[g]) for g in common_genes}
top_genes = sorted(combined, key=combined.get, reverse=True)[:N_GENES]
print(f"  Top-{N_GENES} by min(var_brca, var_crc): selected.")
print(f"  Top-5 example: {top_genes[:5]}")
print(f"  Min var in top-{N_GENES}: {combined[top_genes[-1]]:.4f}")

print("\n[4] Subset both AnnData to fixed gene set...")
ad_brca_sub = ad_brca[:, top_genes].copy()
ad_crc_sub = ad_crc[:, top_genes].copy()
print(f"  BRCA subset: {ad_brca_sub.shape}")
print(f"  CRC  subset: {ad_crc_sub.shape}")


def prepare_pair(seed):
    print(f"\n[seed {seed}] Adapting BRCA (subset)...")
    b_brca = adata_to_tsgnn_tensors(
        ad_brca_sub, K=K_BINS, n_genes=N_GENES, k_neighbors=KNN,
        seed=seed, pseudotime_method="dpt"
    )
    print(f"  BRCA adapted: X_seq {b_brca['X_seq'].shape} edges {b_brca['edge_index'].shape[1]}  genes={len(b_brca['gene_names'])}")
    print(f"[seed {seed}] Adapting CRC (subset)...")
    b_crc = adata_to_tsgnn_tensors(
        ad_crc_sub, K=K_BINS, n_genes=N_GENES, k_neighbors=KNN,
        seed=seed, pseudotime_method="dpt"
    )
    print(f"  CRC  adapted: X_seq {b_crc['X_seq'].shape} edges {b_crc['edge_index'].shape[1]}  genes={len(b_crc['gene_names'])}")

    # Force common gene ordering (handle TP53 force-include + reordering)
    intersect = sorted(set(b_brca["gene_names"]) & set(b_crc["gene_names"]))
    print(f"  Final gene intersection after adapter: {len(intersect)}")
    b_brca_aligned = align_to_gene_set(b_brca, intersect)
    b_crc_aligned = align_to_gene_set(b_crc, intersect)
    return b_brca_aligned, b_crc_aligned, len(intersect)


def run_one(name, seed, cfg, b_brca_aligned, b_crc_aligned):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_brca = b_brca_aligned["X_seq"].to(device)
    ei_brca = b_brca_aligned["edge_index"].to(device)
    X_crc = b_crc_aligned["X_seq"].to(device)
    ei_crc = b_crc_aligned["edge_index"].to(device)
    K, N, _ = X_brca.shape
    if X_crc.shape[1] != N:
        raise RuntimeError(f"shape mismatch BRCA N={N} vs CRC N={X_crc.shape[1]}")
    esm = torch.randn(1280, device=device) * 0.1

    base = dict(
        num_nodes=N, num_edges=ei_brca.shape[1],
        input_dim=1, esm_dim=1280, conditioning_dim=128,
        edge_index=ei_brca, num_diffusion_steps=3,
        use_allele_conditioning=True,
        use_vectorized=True,
        use_normalized_laplacian=True,
    )
    base.update(cfg)
    torch.manual_seed(seed)
    model = TSGNN(**base).to(device)
    loss = train_model(model, X_brca, esm)

    res_brca = eval_model(model, X_brca, esm)
    res_crc = eval_model(model, X_crc, esm, edge_index=ei_crc)

    print(f"  [{name} seed={seed}]  loss={loss:.4f}  "
          f"BRCA r_abs={res_brca['r_abs']:+.4f} r_d={res_brca['r_delta']:+.4f}  "
          f"CRC r_abs={res_crc['r_abs']:+.4f} r_d={res_crc['r_delta']:+.4f}")
    return res_brca, res_crc


print("\n[5] Running BRCA→CRC zero-shot probe (chosen + baseline) over 3 seeds...")

chosen_results = []
baseline_results = []
n_common = None

for seed in SEEDS:
    b_brca_aligned, b_crc_aligned, n_common = prepare_pair(seed)
    rb_c, rc_c = run_one("chosen  ", seed, CHOSEN, b_brca_aligned, b_crc_aligned)
    rb_b, rc_b = run_one("baseline", seed, BASELINE, b_brca_aligned, b_crc_aligned)
    chosen_results.append((rb_c, rc_c))
    baseline_results.append((rb_b, rc_b))


def summary(name, rows):
    brca_d = [r[0]["r_delta"] for r in rows]
    crc_d = [r[1]["r_delta"] for r in rows]
    crc_a = [r[1]["r_abs"] for r in rows]
    brca_a = [r[0]["r_abs"] for r in rows]
    print(f"  {name:<10}  BRCA r_a:{np.mean(brca_a):+.3f} r_d:{np.mean(brca_d):+.3f}±{np.std(brca_d):.3f}   "
          f"CRC r_a:{np.mean(crc_a):+.3f} r_d:{np.mean(crc_d):+.3f}±{np.std(crc_d):.3f}   "
          f"transfer_gap:{np.mean(brca_d)-np.mean(crc_d):+.3f}")
    return np.mean(crc_d), np.std(crc_d)


print(f"\n=== Cross-Dataset Results v2 (n_common={n_common} genes, 3 seeds) ===")
chosen_mean, chosen_std = summary("chosen", chosen_results)
baseline_mean, baseline_std = summary("baseline", baseline_results)

paired_delta = [chosen_results[i][1]["r_delta"] - baseline_results[i][1]["r_delta"] for i in range(len(SEEDS))]
print(f"\n[paired] chosen vs baseline (CRC):  Δr_d={np.mean(paired_delta):+.4f}  ({sum(d>0 for d in paired_delta)}/3 wins chosen)")

print(f"\nVerdict (BRCA→CRC zero-shot, n={n_common} genes):")
if n_common < 100:
    print(f"  INVALID: n_common={n_common} too small. Increase combined HVG pool.")
elif chosen_mean > 0.30:
    print(f"  GO: chosen achieves CRC r_delta={chosen_mean:.4f} > 0.30. Sheaf framework transfers. Paper viable.")
elif chosen_mean > 0.20:
    print(f"  PARTIAL: chosen CRC r_delta={chosen_mean:.4f} in [0.20, 0.30]. Mention as secondary result.")
else:
    print(f"  NO-GO: chosen CRC r_delta={chosen_mean:.4f} < 0.20. Reframe scope or rebuild.")
