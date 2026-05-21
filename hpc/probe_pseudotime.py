"""Probe adjacent-timestep correlation to test MLP-identity hypothesis.

If pseudotime binning makes X[K-2] ≈ X[K-1], MLP trivially beats everyone
by learning identity. Expected: high correlation between adjacent bins.
"""
import sys, os
sys.path.insert(0, "/home/3393519/ts-gnn/src")
import torch, numpy as np, yaml
from tsgnn.data.registry import load_dataset
from tsgnn.data.nature_adapter import adata_to_tsgnn_tensors

cfg = yaml.safe_load(open("/home/3393519/ts-gnn/configs/nature.yaml"))
print("=== Probe pseudotime-adjacency correlation ===")

def probe(tag, X):
    print(f"{tag} X shape:", X.shape)
    for t in range(X.shape[0]-1):
        c = np.corrcoef(X[t], X[t+1])[0,1]
        print(f"  {tag} corr(X[{t}], X[{t+1}])={c:.4f}")
    print(f"  {tag} corr(X[-2], X[-1]) =",
          float(np.corrcoef(X[-2], X[-1])[0,1]))

ad = load_dataset("brca_wu2021")
adc = load_dataset("crc_pelka2021")

for method in ("mean_expr", "dpt", "shuffled"):
    print(f"\n=== method={method} ===")
    b = adata_to_tsgnn_tensors(ad, K=10, n_genes=500, k_neighbors=8, seed=42,
                                pseudotime_method=method)
    probe(f"BRCA[{method}]", b["X_seq"].numpy().squeeze(-1))
    bc = adata_to_tsgnn_tensors(adc, K=10, n_genes=500, k_neighbors=8, seed=42,
                                 ref_mu=b["stats"]["mu"], ref_sd=b["stats"]["sd"],
                                 ref_genes=b["stats"]["genes"],
                                 pseudotime_method=method)
    probe(f"CRC[{method}]", bc["X_seq"].numpy().squeeze(-1))
