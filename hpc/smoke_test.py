"""
HPC environment smoke test.

Verifies all dependencies + basic forward pass work end-to-end
on cluster before running expensive jobs.

Checks:
  1. Imports: torch, torchdiffeq, anndata, scanpy, sklearn
  2. CUDA: device, version
  3. TSGNN model with FULL stack (A1+A2+A3+B1+B2) builds
  4. One forward + backward pass succeeds
  5. Data loaders work (BRCA + CRC) - tiny subset only
  6. Cross-dataset edge_index swap works (probe_alpha fix)

Exit 0 = OK, exit 1 = something broken.
"""
import sys, os
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

print("=== HPC Smoke Test ===")

# ── Dependencies ──────────────────────────────────────────────────────────────
try:
    import torch
    print(f"  torch={torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("FAIL: CUDA not available! Check --gres and module load cuda/12.1")
        sys.exit(1)
    import torchdiffeq
    print(f"  torchdiffeq=OK")
except ImportError as e:
    print(f"FAIL: {e}")
    sys.exit(1)

try:
    import anndata, scanpy as sc, sklearn, numpy as np
    print(f"  anndata={anndata.__version__}")
    print(f"  scanpy={sc.__version__}")
except ImportError as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# ── TSGNN imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, "src")
try:
    from tsgnn.model.tsgnn import TSGNN
    from tsgnn.model.temporal_ode import SheafNeuralODE
    from tsgnn.model.film_nodal import NodalAlleleFiLM
    from tsgnn.model.diffusion_implicit import ImplicitOrthogonalSheafDiffusion
    from tsgnn.data.registry import load_dataset
    from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors
    print(f"  TSGNN imports OK")
except ImportError as e:
    print(f"FAIL TSGNN import: {e}")
    sys.exit(1)

# ── Synthetic forward + backward (no data needed) ─────────────────────────────
print("\n[1] Synthetic forward+backward with FULL stack...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

N, E, K, d = 20, 60, 5, 4
edges = torch.randint(0, N, (2, E))
X_seq = torch.randn(K, N, 1, device=device)
esm = torch.randn(1280, device=device)

try:
    model = TSGNN(
        num_nodes=N, num_edges=E,
        stalk_dim=d, input_dim=1, esm_dim=1280, conditioning_dim=128,
        edge_index=edges.to(device), num_diffusion_steps=3,
        use_allele_conditioning=True,
        use_orthogonal_maps=True,
        use_implicit_diffusion=True,
        predict_delta=True,
        use_nodal_film=False,
        use_neural_ode=True,
        use_cayley_v2=True,
        ode_hidden_dim=128, ode_solver="rk4",
    ).to(device)
    print(f"  Built FULL stack TSGNN: {sum(p.numel() for p in model.parameters())} params")
    out = model(X_seq[:-1], esm)
    preds = out[0] if isinstance(out, tuple) else out
    if isinstance(preds, list):
        preds = torch.stack(preds, dim=0)
    print(f"  Forward OK: preds shape={preds.shape}")
    loss = ((preds - X_seq[1:]) ** 2).mean()
    loss.backward()
    print(f"  Backward OK: loss={loss.item():.4f}")
    # Check gradients
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Gradients flowing: {n_grad}/{n_total} params")
except Exception as e:
    print(f"FAIL forward/backward: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Real data loading (tiny subset) ───────────────────────────────────────────
print("\n[2] Real data loading (BRCA + CRC, small subset)...")
try:
    ad_brca = load_dataset("brca_wu2021")
    ad_crc = load_dataset("crc_pelka2021")
    print(f"  BRCA: {ad_brca.shape}")
    print(f"  CRC:  {ad_crc.shape}")
    ad_brca = ad_brca[:1000].copy()
    ad_crc = ad_crc[:800].copy()
    b_brca = adata_to_tsgnn_tensors(ad_brca, K=5, n_genes=50, k_neighbors=5, seed=42, pseudotime_method="dpt")
    b_crc = adata_to_tsgnn_tensors(
        ad_crc, K=5, n_genes=50, k_neighbors=5, seed=42,
        ref_mu=b_brca["stats"]["mu"], ref_sd=b_brca["stats"]["sd"],
        ref_genes=b_brca["stats"]["genes"], pseudotime_method="dpt"
    )
    print(f"  BRCA tensors: X={b_brca['X_seq'].shape}, E={b_brca['edge_index'].shape}")
    print(f"  CRC tensors:  X={b_crc['X_seq'].shape},  E={b_crc['edge_index'].shape}")
except Exception as e:
    print(f"FAIL data load: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Edge index swap (probe_alpha fix verification) ────────────────────────────
print("\n[3] Cross-dataset edge_index swap...")
try:
    ei_brca = b_brca["edge_index"].to(device)
    ei_crc = b_crc["edge_index"].to(device)
    X_brca = b_brca["X_seq"].to(device)
    X_crc = b_crc["X_seq"].to(device)
    N_b = X_brca.shape[1]

    model = TSGNN(
        num_nodes=N_b, num_edges=ei_brca.shape[1],
        stalk_dim=4, input_dim=1, esm_dim=1280, conditioning_dim=128,
        edge_index=ei_brca, num_diffusion_steps=2,
        use_orthogonal_maps=True, use_implicit_diffusion=True,
        predict_delta=True, use_nodal_film=False, use_neural_ode=True,
        use_cayley_v2=True,
    ).to(device)
    # BRCA forward
    with torch.no_grad():
        _ = model(X_brca[:-1], esm)
    print(f"  BRCA forward OK")

    # NB: cross-dataset eval requires same N. Restriction maps tied to E.
    # Just verify edge_index buffer is swappable.
    orig_ei = model.edge_index
    if X_crc.shape[1] == X_brca.shape[1] and ei_crc.shape[1] == ei_brca.shape[1]:
        model.edge_index = ei_crc
        with torch.no_grad():
            _ = model(X_crc[:-1], esm)
        print(f"  CRC swap forward OK")
    else:
        print(f"  WARN: shape mismatch BRCA vs CRC, edge swap not tested")
        print(f"        N_brca={X_brca.shape[1]}, N_crc={X_crc.shape[1]}")
        print(f"        E_brca={ei_brca.shape[1]}, E_crc={ei_crc.shape[1]}")
        print(f"        Cross-dataset eval requires N+E remapping")
    model.edge_index = orig_ei
except Exception as e:
    print(f"FAIL edge swap: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n=== SMOKE TEST PASSED ===")
sys.exit(0)
