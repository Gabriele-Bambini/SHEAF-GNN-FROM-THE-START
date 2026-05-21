"""
Probe TSGNN-v2 (Cayley + EdgeWeightNet) vs v1 (Legacy QR).
"""
import sys, os
import torch
import numpy as np
import yaml
import logging

# Ensure local src is in path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from tsgnn.data.registry import load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors
from tsgnn.model.tsgnn import create_tsgnn_from_config
from tsgnn.model.baselines import MLPBaseline

print("=== Probe v2 (Cayley) vs v1 (Legacy) ===")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_EPOCHS = 50
torch.manual_seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1] Loading BRCA + CRC...")
ad_brca = load_dataset("brca_wu2021")
ad_crc  = load_dataset("crc_pelka2021")

# Force gene intersection
common_genes = sorted(list(set(ad_brca.var_names) & set(ad_crc.var_names)))
ad_brca = ad_brca[:, common_genes].copy()
ad_crc  = ad_crc[:, common_genes].copy()

b_brca = adata_to_tsgnn_tensors(
    ad_brca, K=10, n_genes=500, k_neighbors=8, seed=SEED,
    pseudotime_method="dpt"
)
b_crc = adata_to_tsgnn_tensors(
    ad_crc, K=10, n_genes=500, k_neighbors=8, seed=SEED,
    ref_mu=b_brca["stats"]["mu"],
    ref_sd=b_brca["stats"]["sd"],
    ref_genes=b_brca["stats"]["genes"],
    pseudotime_method="dpt"
)

X_brca = b_brca["X_seq"].to(device)
X_crc  = b_crc["X_seq"].to(device)
ei_brca = b_brca["edge_index"].to(device)
ei_crc  = b_crc["edge_index"].to(device)
N = X_brca.shape[1]

print(f"  BRCA: N={N}, E={ei_brca.shape[1]}")
print(f"  CRC:  N={X_crc.shape[1]}, E={ei_crc.shape[1]}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8: return 0.0
    return float(np.corrcoef(a, b)[0,1])

def train_model(model, X_full, esm_vec, n_epochs=N_EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    K = X_full.shape[0]
    X_train = X_full[:-2]
    Y_train = X_full[1:-1]
    model.train()
    for ep in range(n_epochs):
        opt.zero_grad()
        out = model(X_train, esm_vec)
        preds = torch.stack(out[0], dim=0)
        loss = ((preds - Y_train) ** 2).mean()
        loss.backward()
        opt.step()
    return loss.item()

def residualized_delta_corr(
    pred_delta: torch.Tensor,
    true_delta: torch.Tensor,
    train_delta_mean=None,
):
    pred = pred_delta.detach().float()
    true = true_delta.detach().float()

    pred_flat = pred.flatten()
    true_flat = true.flatten()
    valid = torch.isfinite(pred_flat) & torch.isfinite(true_flat)
    if valid.sum() < 3:
        return {"r_delta": 0.0, "r_delta_perp": 0.0}

    pf, tf = pred_flat[valid], true_flat[valid]
    r_raw = float(torch.corrcoef(torch.stack([pf, tf]))[0, 1].item())

    if train_delta_mean is not None:
        m = train_delta_mean.detach().float().flatten()
    else:
        m = true.mean(0).flatten()

    m_norm = m / (m.norm() + 1e-8)

    pred_proj = (pred @ m_norm.unsqueeze(-1)) * m_norm
    true_proj = (true @ m_norm.unsqueeze(-1)) * m_norm

    pred_perp = (pred - pred_proj).flatten()
    true_perp = (true - true_proj).flatten()

    valid2 = torch.isfinite(pred_perp) & torch.isfinite(true_perp)
    if valid2.sum() < 3:
        return {"r_delta": r_raw, "r_delta_perp": 0.0}

    pp, tp = pred_perp[valid2], true_perp[valid2]
    r_perp = float(torch.corrcoef(torch.stack([pp, tp]))[0, 1].item())

    return {
        "r_delta":      r_raw,
        "r_delta_perp": r_perp,
    }

def eval_model(model, X_full, esm_vec, edge_index=None):
    model.eval()
    if edge_index is not None:
        orig_ei = model.edge_index
        model.edge_index = edge_index
    with torch.no_grad():
        out = model(X_full[:-1], esm_vec)
        preds = torch.stack(out[0], dim=0)
    if edge_index is not None:
        model.edge_index = orig_ei
        
    p_last = preds[-1].cpu()
    x_last = X_full[-1].cpu()
    x_prev = X_full[-2].cpu()
    
    pred_delta = p_last - x_prev
    true_delta = x_last - x_prev
    
    res = residualized_delta_corr(pred_delta, true_delta)
    
    r_abs = pearson(p_last.numpy(), x_last.numpy())
    return {"r_abs": r_abs, "r_delta_raw": res["r_delta"], "r_delta_perp": res["r_delta_perp"]}

esm = torch.randn(1280, device=device) * 0.1

# ── Runs ──────────────────────────────────────────────────────────────────────
print("\n[2a] Config A: TSGNN-v2 (Cayley)...")
cfg_v2 = yaml.safe_load(open("configs/nature_v2.yaml"))
model_A = create_tsgnn_from_config(cfg_v2, ei_brca).to(device)
loss_A = train_model(model_A, X_brca, esm)
brca_A = eval_model(model_A, X_brca, esm)
crc_A = eval_model(model_A, X_crc, esm, edge_index=ei_crc)

print("\n[2b] Config B: TSGNN-v1 (Legacy)...")
cfg_v1 = yaml.safe_load(open("configs/nature_v1.yaml"))
model_B = create_tsgnn_from_config(cfg_v1, ei_brca).to(device)
loss_B = train_model(model_B, X_brca, esm)
brca_B = eval_model(model_B, X_brca, esm)
crc_B = eval_model(model_B, X_crc, esm, edge_index=ei_crc)

print("\n[2c] Config C: MLP Baseline...")
model_C = MLPBaseline(num_nodes=N, input_dim=1, esm_dim=1280, hidden_dim=256, output_dim=1).to(device)
loss_C = train_model(model_C, X_brca, esm)
brca_C = eval_model(model_C, X_brca, esm)
crc_C = eval_model(model_C, X_crc, esm)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"{'Metric':<20} {'v2 (Cayley)':>15} {'v1 (Legacy)':>15} {'MLP':>15}")
for label, mA, mB, mC in [
    ("BRCA r_abs", brca_A, brca_B, brca_C), 
    ("BRCA r_delta_perp", brca_A, brca_B, brca_C),
    ("CRC r_abs", crc_A, crc_B, crc_C), 
    ("CRC r_delta_perp", crc_A, crc_B, crc_C)
]:
    key = label.split()[-1]
    vA, vB, vC = mA[key], mB[key], mC[key]
    print(f"  {label:<18} {vA:>15.4f} {vB:>15.4f} {vC:>15.4f}")
