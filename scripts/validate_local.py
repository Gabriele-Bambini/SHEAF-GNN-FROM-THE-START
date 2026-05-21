"""
Local validation: A1+A3+B1+B2 impact on BRCA→CRC transfer.

Quick local benchmark:
  - Train A1+A3+B1+B2 stack on BRCA (K=10, n_genes=100, N_epochs=30)
  - Eval same-dataset BRCA: r_delta
  - Eval cross-dataset CRC: r_delta (critical for P1)
  - Compare vs baseline (no A1, no A3, no B1, no B2, d=8)

Exit code 0 = A outperforms B on CRC r_delta (P1-relevant)
Exit code 1 = B better or unclear
"""
import sys, os
sys.path.insert(0, "src")

import torch
import numpy as np
import yaml
import logging
logging.basicConfig(level=logging.WARNING)

from tsgnn.data.registry import load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors
from tsgnn.model.tsgnn import TSGNN

print("=== Local Validation: A1+A3+B1+B2 vs Baseline ===")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_EPOCHS = 30
torch.manual_seed(SEED)

# Load data
print("\n[1] Loading BRCA + CRC (small subset for speed)...")
ad_brca = load_dataset("brca_wu2021")
ad_crc = load_dataset("crc_pelka2021")

# Subset to speed up (use AnnData slicing)
ad_brca = ad_brca[:2000, :500]  # 2000 cells, 500 genes
ad_crc = ad_crc[:1500, :500]

b_brca = adata_to_tsgnn_tensors(
    ad_brca, K=10, n_genes=100, k_neighbors=8, seed=SEED,
    pseudotime_method="dpt"
)
b_crc = adata_to_tsgnn_tensors(
    ad_crc, K=10, n_genes=100, k_neighbors=8, seed=SEED,
    ref_mu=b_brca["stats"]["mu"],
    ref_sd=b_brca["stats"]["sd"],
    ref_genes=b_brca["stats"]["genes"],
    pseudotime_method="dpt"
)

X_brca = b_brca["X_seq"].to(device)
X_crc = b_crc["X_seq"].to(device)
ei_brca = b_brca["edge_index"].to(device)
ei_crc = b_crc["edge_index"].to(device)
K, N, _ = X_brca.shape

print(f"  BRCA: K={K}, N={N}, E={ei_brca.shape[1]}")
print(f"  CRC:  K={X_crc.shape[0]}, N={X_crc.shape[1]}, E={ei_crc.shape[1]}")

# Fake ESM
torch.manual_seed(SEED)
esm = torch.randn(1280, device=device) * 0.1

def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def eval_model(model, X_full, esm_vec, edge_index=None):
    """r_delta = corr(delta_pred[-1], delta_true[-1])."""
    model.eval()
    with torch.no_grad():
        if edge_index is not None:
            orig_ei = model.edge_index
            model.edge_index = edge_index
        try:
            out = model(X_full[:-1], esm_vec)
            preds = out[0] if isinstance(out, tuple) else out
            if isinstance(preds, list):
                preds = torch.stack(preds, dim=0)
        finally:
            if edge_index is not None:
                model.edge_index = orig_ei

    p_last = preds[-1].cpu().numpy().squeeze()
    x_last = X_full[-1].cpu().numpy().squeeze()
    x_prev = X_full[-2].cpu().numpy().squeeze()
    delta_pred = p_last - x_prev
    delta_true = x_last - x_prev
    r_abs = pearson(p_last, x_last)
    r_delta = pearson(delta_pred, delta_true)
    return {"r_abs": r_abs, "r_delta": r_delta}

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

# Config A: A1+A3+B1+B2 (new stack)
print("\n[2a] Training A1+A3+B1+B2+d=4...")
model_A = TSGNN(
    num_nodes=N, num_edges=ei_brca.shape[1],
    stalk_dim=4, input_dim=1, esm_dim=1280, conditioning_dim=128,
    edge_index=ei_brca, num_diffusion_steps=3,
    use_allele_conditioning=True, use_vectorized=True,
    use_normalized_laplacian=True,
    use_orthogonal_maps=True,
    use_implicit_diffusion=True,
    predict_delta=True,
    use_nodal_film=True,
    use_neural_ode=True,
    ode_hidden_dim=256, ode_solver="rk4",
).to(device)

torch.manual_seed(SEED)
loss_A = train_model(model_A, X_brca, esm)
brca_A = eval_model(model_A, X_brca, esm)
crc_A = eval_model(model_A, X_crc, esm, edge_index=ei_crc)
print(f"  train_loss={loss_A:.4f}")
print(f"  BRCA: r_abs={brca_A['r_abs']:.4f}  r_delta={brca_A['r_delta']:.4f}")
print(f"  CRC:  r_abs={crc_A['r_abs']:.4f}   r_delta={crc_A['r_delta']:.4f}")

# Config B: baseline (no A1, A3, B1, B2, d=8)
print("\n[2b] Training baseline (no A1, no A3, no B1, no B2, d=8)...")
model_B = TSGNN(
    num_nodes=N, num_edges=ei_brca.shape[1],
    stalk_dim=8, input_dim=1, esm_dim=1280, conditioning_dim=128,
    edge_index=ei_brca, num_diffusion_steps=3,
    use_allele_conditioning=False,
    use_vectorized=True,
    use_normalized_laplacian=True,
    use_orthogonal_maps=False,
    use_implicit_diffusion=False,
    predict_delta=False,
    use_nodal_film=False,
    use_neural_ode=False,
).to(device)

torch.manual_seed(SEED)
loss_B = train_model(model_B, X_brca, esm)
brca_B = eval_model(model_B, X_brca, esm)
crc_B = eval_model(model_B, X_crc, esm, edge_index=ei_crc)
print(f"  train_loss={loss_B:.4f}")
print(f"  BRCA: r_abs={brca_B['r_abs']:.4f}  r_delta={brca_B['r_delta']:.4f}")
print(f"  CRC:  r_abs={crc_B['r_abs']:.4f}   r_delta={crc_B['r_delta']:.4f}")

# Summary
print("\n=== SUMMARY ===")
print(f"{'Metric':<25} {'A1+A3+B1+B2':>18} {'Baseline':>18} {'Delta':>8}")
print("-" * 75)
metrics = [
    ("BRCA r_abs", brca_A["r_abs"], brca_B["r_abs"]),
    ("BRCA r_delta", brca_A["r_delta"], brca_B["r_delta"]),
    ("CRC r_abs", crc_A["r_abs"], crc_B["r_abs"]),
    ("CRC r_delta", crc_A["r_delta"], crc_B["r_delta"]),
]
for name, vA, vB in metrics:
    delta = vA - vB
    sign = "+" if delta >= 0 else ""
    print(f"  {name:<23} {vA:>18.4f} {vB:>18.4f} {sign}{delta:>7.4f}")

print("\nP1 threshold (CRC r_delta): 0.35")
a_passes = crc_A["r_delta"] > 0.35
b_passes = crc_B["r_delta"] > 0.35
print(f"  A passes: {a_passes} (r={crc_A['r_delta']:.4f})")
print(f"  B passes: {b_passes} (r={crc_B['r_delta']:.4f})")

# Exit code: 0 if A better on CRC, 1 otherwise
exit_code = 0 if crc_A["r_delta"] > crc_B["r_delta"] else 1
result_str = "PASS: A better" if exit_code == 0 else "FAIL: B better"
print(f"\nExit code: {exit_code} ({result_str})")
sys.exit(exit_code)
