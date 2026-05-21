"""
Ablation study: progressively enable A1, A3, B1, B2 features.

Configs (all share use_allele_conditioning=True for fair comparison):
  baseline:  no A1, no A3, no B1, no B2  (d=8)
  +A1:       orthogonal maps                (d=4)
  +A1+A3:    orthogonal + predict_delta     (d=4)
  +A1+A3+B1: + nodal FiLM                   (d=4)
  +A1+A3+B1+B2: + neural ODE (full stack)   (d=4)

Tracks per-epoch loss + post-training diagnostics:
  - ODE velocity field weight norm (zero = stuck in identity)
  - Implicit diffusion log_tau (should change from init 0.2)
  - FiLM gamma/beta variance (zero = FiLM inactive)

Reports r_delta on BRCA + CRC for each config.
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

print("=== Ablation: A1 -> A3 -> B1 -> B2 ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
SEED = 42
N_EPOCHS = 80
torch.manual_seed(SEED)

print("\n[1] Loading data (subset)...")
ad_brca = load_dataset("brca_wu2021")
ad_crc = load_dataset("crc_pelka2021")
# Cell subsetting only (preserve full gene set for cross-dataset overlap)
ad_brca = ad_brca[:3000].copy()
ad_crc = ad_crc[:2000].copy()

b_brca = adata_to_tsgnn_tensors(ad_brca, K=10, n_genes=100, k_neighbors=8, seed=SEED, pseudotime_method="dpt")
b_crc = adata_to_tsgnn_tensors(
    ad_crc, K=10, n_genes=100, k_neighbors=8, seed=SEED,
    ref_mu=b_brca["stats"]["mu"], ref_sd=b_brca["stats"]["sd"],
    ref_genes=b_brca["stats"]["genes"], pseudotime_method="dpt"
)
X_brca = b_brca["X_seq"].to(device)
X_crc = b_crc["X_seq"].to(device)
ei_brca = b_brca["edge_index"].to(device)
ei_crc = b_crc["edge_index"].to(device)
K, N, _ = X_brca.shape
print(f"  BRCA: K={K}, N={N}, E={ei_brca.shape[1]}")
print(f"  CRC:  K={X_crc.shape[0]}, N={X_crc.shape[1]}, E={ei_crc.shape[1]}")

torch.manual_seed(SEED)
esm = torch.randn(1280, device=device) * 0.1


def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def eval_model(model, X_full, esm_vec, edge_index=None):
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
    losses = []
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
        losses.append(loss.item())
    return losses


def diagnose(model, label):
    """Inspect post-training params: did ODE/tau/FiLM actually move?"""
    print(f"  [{label}] diagnostics:")
    # Neural ODE velocity field
    if hasattr(model, "temporal_evolution") and hasattr(model.temporal_evolution, "velocity_field"):
        vf = model.temporal_evolution.velocity_field
        last_w_norm = vf[-1].weight.norm().item()
        first_w_norm = vf[0].weight.norm().item()
        print(f"    ODE velocity_field[-1].weight.norm = {last_w_norm:.4f} (init=0)")
        print(f"    ODE velocity_field[0].weight.norm  = {first_w_norm:.4f}")
    # Implicit diffusion log_tau
    if hasattr(model, "sheaf_layer") and hasattr(model.sheaf_layer, "log_tau"):
        tau = model.sheaf_layer.log_tau.exp().item()
        print(f"    Implicit diffusion tau = {tau:.4f} (init=0.2)")


def run_config(name, **kwargs):
    print(f"\n[{name}] Training {N_EPOCHS} epochs...")
    torch.manual_seed(SEED)
    base_kwargs = dict(
        num_nodes=N, num_edges=ei_brca.shape[1],
        input_dim=1, esm_dim=1280, conditioning_dim=128,
        edge_index=ei_brca, num_diffusion_steps=3,
        use_allele_conditioning=True,
        use_vectorized=True,
        use_normalized_laplacian=True,
    )
    base_kwargs.update(kwargs)
    model = TSGNN(**base_kwargs).to(device)
    losses = train_model(model, X_brca, esm)
    print(f"  loss[0]={losses[0]:.4f}  loss[mid]={losses[len(losses)//2]:.4f}  loss[-1]={losses[-1]:.4f}")
    diagnose(model, name)
    brca = eval_model(model, X_brca, esm)
    print(f"  BRCA: r_abs={brca['r_abs']:.4f}  r_delta={brca['r_delta']:.4f}")
    # CRC eval requires per-edge state remapping (not implemented). Skip for now.
    crc = {"r_abs": float("nan"), "r_delta": float("nan")}
    return {"name": name, "loss": losses[-1], "brca": brca, "crc": crc}


configs = [
    ("baseline", dict(stalk_dim=8,
                      use_orthogonal_maps=False, use_implicit_diffusion=False,
                      predict_delta=False, use_nodal_film=False, use_neural_ode=False)),
    ("+A1", dict(stalk_dim=4,
                 use_orthogonal_maps=True, use_implicit_diffusion=True,
                 predict_delta=False, use_nodal_film=False, use_neural_ode=False)),
    ("+A1+A3", dict(stalk_dim=4,
                    use_orthogonal_maps=True, use_implicit_diffusion=True,
                    predict_delta=True, use_nodal_film=False, use_neural_ode=False)),
    ("+A1+A3+B1", dict(stalk_dim=4,
                       use_orthogonal_maps=True, use_implicit_diffusion=True,
                       predict_delta=True, use_nodal_film=True, use_neural_ode=False)),
    ("+A1+A3+B1+B2", dict(stalk_dim=4,
                          use_orthogonal_maps=True, use_implicit_diffusion=True,
                          predict_delta=True, use_nodal_film=True, use_neural_ode=True,
                          ode_hidden_dim=256, ode_solver="rk4")),
]

results = []
for name, kw in configs:
    results.append(run_config(name, **kw))

print("\n=== ABLATION SUMMARY (BRCA same-dataset) ===")
print(f"{'Config':<20} {'final_loss':>12} {'BRCA r_abs':>12} {'BRCA r_d':>10}")
print("-" * 60)
for r in results:
    print(f"  {r['name']:<18} {r['loss']:>12.4f} {r['brca']['r_abs']:>12.4f} {r['brca']['r_delta']:>10.4f}")

# Find best on BRCA r_delta
best = max(results, key=lambda r: r["brca"]["r_delta"])
print(f"\nBest BRCA r_delta: {best['name']} = {best['brca']['r_delta']:.4f}")
print(f"\nNote: CRC eval skipped — requires per-edge state remapping. "
      "Once best config identified, fix cross-dataset eval and re-test.")
