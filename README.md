# TS-GNN: Temporal Sheaf Graph Neural Network

Allele-conditioned gene regulatory network rewiring via temporal sheaf diffusion on pseudotime-ordered single-cell transcriptomes.

## Overview

TS-GNN fuses three components:
1. **Neural Sheaf Diffusion** (Bodnar et al., NeurIPS 2022) — connection Laplacian with d x d restriction maps encoding activating/repressive regulation
2. **GRU Temporal Evolution** — shared GRU evolving restriction maps over pseudotime
3. **FiLM Allele Conditioning** — ESM-2 protein embeddings modulating regulatory rewiring per TP53 mutation

## Quick Start

### Installation

```bash
pip install -e ".[all]"
```

### One-Command Pipeline

```bash
# Full pipeline (downloads data, trains, evaluates)
python scripts/run_pipeline.py --config configs/default.yaml

# Skip download (use existing data)
python scripts/run_pipeline.py --config configs/default.yaml --skip-download

# CRC proof-of-concept
python scripts/run_pipeline.py --config configs/phase0_crc.yaml --skip-download
```

### Docker

```bash
docker build -t tsgnn .
docker run --gpus all tsgnn --config configs/default.yaml
```

## Project Structure

```
ts-gnn/
├── configs/
│   ├── default.yaml          # Full hyperparameter configuration
│   └── phase0_crc.yaml       # CRC proof-of-concept overrides
├── src/tsgnn/
│   ├── data/
│   │   ├── download.py       # Data acquisition (GSE178341, JASPAR, STRING, DepMap)
│   │   ├── allele.py         # ESM-2 embeddings for TP53 alleles
│   │   ├── preprocess.py     # scRNA-seq preprocessing pipeline
│   │   ├── grn_construction.py  # Multi-layer GRN construction
│   │   └── temporal.py       # Pseudotime inference and temporal binning
│   ├── model/
│   │   ├── sheaf.py          # Neural Sheaf Diffusion layer
│   │   ├── temporal.py       # GRU temporal evolution
│   │   ├── film.py           # FiLM allele conditioning
│   │   ├── tsgnn.py          # Full TS-GNN architecture
│   │   └── baselines.py      # 5 baseline models
│   ├── training/
│   │   ├── loss.py           # 4-term composite loss
│   │   └── trainer.py        # Training loop with early stopping
│   ├── evaluation/
│   │   ├── metrics.py        # 5 evaluation metrics
│   │   ├── ablation.py       # Ablation study runner
│   │   └── benchmarks.py     # Baseline comparison
│   └── visualization/
│       ├── rewiring.py       # Rewiring trajectory plots
│       ├── attention.py      # Restriction map analysis
│       └── networks.py       # GRN and embedding visualization
├── scripts/
│   ├── run_pipeline.py       # End-to-end pipeline
│   ├── run_ablations.py      # Ablation studies
│   └── run_benchmarks.py     # Baseline comparisons
├── tests/                    # Unit tests
├── Dockerfile
└── pyproject.toml
```

## Architecture

```
For each time step t:
1. Project input features:     x = proj(X(t))           [N, input_dim] -> [N, d]
2. Evolve restriction maps:    R(t+1) = GRU(R(t), ctx)  [E, 2, d, d]
3. FiLM conditioning:          R' = gamma * R + beta     (from ESM-2 embedding)
4. Connection Laplacian:       L_F = sheaf_laplacian(R') [Nd, Nd]
5. Sheaf diffusion:            x = diffuse(x, L_F)      (k steps)
6. Project to expression:      X_hat = proj_out(x)       [N, d] -> [N, input_dim]
```

## Loss Function

```
L = L_expr + lambda_1 * L_topo + lambda_2 * L_regulon + lambda_3 * L_sparse

L_expr:    MSE between predicted and observed gene expression
L_topo:    Hinge penalty on excessive edge weight changes (Fused Lasso analog)
L_regulon: Negative Spearman correlation with VIPER TF activity
L_sparse:  L1 norm of dL_F/dz_allele (allele-specific perturbation sparsity)
```

## Evaluation Metrics

1. Cross-cancer zero-shot generalization (Pearson r)
2. Allele-specific edge enrichment (Fisher exact test)
3. Chromatin remodeler concordance (ChIP-seq overlap)
4. DepMap genetic dependency correlation (Spearman rho)
5. Rewiring distinguishability (permutation test)

## Hardware Requirements

- Training: Single GPU with 16+ GB VRAM (A100 40GB recommended)
- Estimated training time: 4-8 hours on A100
- CPU-only mode supported for development

## Configuration

All hyperparameters are in `configs/default.yaml`:
- `model.stalk_dim`: Sheaf stalk dimension d (default: 4)
- `model.num_diffusion_steps`: Diffusion iterations (default: 3)
- `loss.lambda_1/2/3`: Loss term weights
- `training.lr`: Learning rate (default: 1e-3)
- `training.patience`: Early stopping patience (default: 20)
