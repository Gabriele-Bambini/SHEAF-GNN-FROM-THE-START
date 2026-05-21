"""
Diagnostic probe v2 — trace signal propagation through FiLM → output.

Root cause investigation:
- T5 shows: FiLM γ/β responsive to ESM (Δ ~0.05 L2)
- But: model r_d unchanged (swing 0.001)
→ Signal lost somewhere between FiLM output and final prediction

Trace: ESM → FiLM → R_modulated → L_F → diffusion → stalk → output
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

print("=== Diagnostic Probe v2 — Signal Propagation Trace ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N_GENES, K_BINS, KNN = 500, 10, 8
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

# Load
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

# Train
print("\n[train] Model on BRCA (50 epochs)...")
torch.manual_seed(42)
base = dict(num_nodes=X_b.shape[1], num_edges=ei_b.shape[1], input_dim=1, esm_dim=1280,
            conditioning_dim=128, edge_index=ei_b, num_diffusion_steps=3,
            use_allele_conditioning=True, use_vectorized=True, use_normalized_laplacian=True)
base.update(CHOSEN)
model = TSGNN(**base).to(device)
esm_fake = torch.randn(1280, device=device) * 0.1
train_model(model, X_b, esm_fake, n_epochs=50)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL TRACE: FiLM → R_modulated → Laplacian → stalks → predictions
# ─────────────────────────────────────────────────────────────────────────────

print("\n[SIGNAL TRACE] Capturing intermediate activations")

# Hooks to capture values at each stage
trace_zero = {}
trace_rand = {}

def hook_film(name):
    def h(module, input, output):
        trace_zero[f"{name}_film"] = output.detach().cpu().numpy()
    return h

def hook_laplacian(name):
    def h(module, input, output):
        trace_zero[f"{name}_laplacian"] = output.detach().cpu().numpy()
    return h

def hook_stalks(name):
    def h(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        trace_zero[f"{name}_stalks"] = output.detach().cpu().numpy()
    return h

# Register hooks (only for zero ESM case, we'll manually copy for rand)
model.allele_film.register_forward_hook(hook_film("allele"))

# Run inference with zero ESM, capture everything
model.eval()
with torch.no_grad():
    esm_zero = torch.zeros(1280, device=device)
    out_zero = model(X_c[:-1], esm_zero)
    preds_zero = out_zero[0] if isinstance(out_zero, tuple) else out_zero
    if isinstance(preds_zero, list):
        preds_zero = torch.stack(preds_zero, dim=0)

# Store traces
trace_zero["pred"] = preds_zero.detach().cpu().numpy()

# Manually extract post-diffusion stalks by hooking the interpreter
stalks_zero = None
def capture_stalks_zero(module, input, output):
    global stalks_zero
    stalks_zero = input[0].detach().clone()

model.interpreter.register_forward_hook(capture_stalks_zero)

with torch.no_grad():
    out_zero = model(X_c[:-1], esm_zero)

# Repeat with random ESM
trace_rand["pred"] = None
stalks_rand = None

def capture_stalks_rand(module, input, output):
    global stalks_rand
    stalks_rand = input[0].detach().clone()

model.interpreter.register_forward_hook(capture_stalks_rand)

with torch.no_grad():
    torch.manual_seed(0)
    esm_rand = torch.randn(1280, device=device) * 0.1
    out_rand = model(X_c[:-1], esm_rand)
    preds_rand = out_rand[0] if isinstance(out_rand, tuple) else out_rand
    if isinstance(preds_rand, list):
        preds_rand = torch.stack(preds_rand, dim=0)

trace_rand["pred"] = preds_rand.detach().cpu().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ANALYSIS] Signal preservation through pipeline")

if stalks_zero is not None and stalks_rand is not None:
    stalks_zero_np = stalks_zero.cpu().numpy()
    stalks_rand_np = stalks_rand.cpu().numpy()

    stalk_l2_dist = np.linalg.norm(stalks_zero_np - stalks_rand_np)
    print(f"\nStalks (post-diffusion):")
    print(f"  zero mean={stalks_zero_np.mean():.6f}, std={stalks_zero_np.std():.6f}")
    print(f"  rand mean={stalks_rand_np.mean():.6f}, std={stalks_rand_np.std():.6f}")
    print(f"  Δ L2 dist = {stalk_l2_dist:.6f}")

    if stalk_l2_dist < 1e-3:
        print(f"  ⚠ Stalks converged to same value despite different ESM")
        print(f"     → Diffusion/Laplacian washes out FiLM signal")
    else:
        print(f"  ✓ Stalks differ → signal preserved through diffusion")

pred_zero = trace_zero["pred"]
pred_rand = trace_rand["pred"]
pred_l2_dist = np.linalg.norm(pred_zero - pred_rand)
print(f"\nPredictions (final output):")
print(f"  zero mean={pred_zero.mean():.6f}, std={pred_zero.std():.6f}")
print(f"  rand mean={pred_rand.mean():.6f}, std={pred_rand.std():.6f}")
print(f"  Δ L2 dist = {pred_l2_dist:.6f}")

x_c_last = X_c[-1].cpu().numpy().squeeze()
x_c_prev = X_c[-2].cpu().numpy().squeeze()
p_zero_last = pred_zero[-1].squeeze()
p_rand_last = pred_rand[-1].squeeze()

r_zero = pearson(p_zero_last - x_c_prev, x_c_last - x_c_prev)
r_rand = pearson(p_rand_last - x_c_prev, x_c_last - x_c_prev)

print(f"\nr_delta:")
print(f"  zero: {r_zero:+.4f}")
print(f"  rand: {r_rand:+.4f}")
print(f"  Δ: {r_zero - r_rand:+.6f}")

# Summary logic
print("\n[VERDICT]")
if stalk_l2_dist < 1e-3:
    print("  FiLM-responsive γ/β → but stalks identical")
    print("  ⇒ PROBLEM: Diffusion normalizes FiLM scaling")
    print("    Suspects: (1) L_F normalization in Implicit Diffusion")
    print("             (2) Convergence to same fixed point")
    print("             (3) Edge weights ignored in Laplacian")
elif pred_l2_dist < 1e-3:
    print("  Stalks differ → but predictions identical")
    print("  ⇒ PROBLEM: Output projection erases stalk differences")
    print("    Suspect: 4-dim stalk bottleneck loses allele info")
else:
    print("  ✓ Signal preserved. But r_delta same → other factor.")
    print("    Investigate: are structure differences orthogonal to r_delta metric?")

print("\n=== END PROBE v2 ===")
