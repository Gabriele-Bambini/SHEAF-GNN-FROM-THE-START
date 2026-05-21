#!/usr/bin/env python
"""
Optuna hyperparameter search for TS-GNN.

Runs an adaptive search over learning rate, stalk dimension, FiLM conditioning dim,
number of diffusion steps, loss weights (λ₁, λ₂, λ₃), and weight decay.

Usage::

    python scripts/run_hparam_search.py \
        --config configs/default.yaml \
        --n-trials 50 \
        --n-epochs 30 \
        --out-config configs/optuna_best.yaml \
        --storage sqlite:///optuna_tsgnn.db

    # Resume a previous search from SQLite:
    python scripts/run_hparam_search.py \
        --n-trials 100 \
        --storage sqlite:///optuna_tsgnn.db

The script loads the same data pipeline as run_pipeline.py so it can be run
immediately after download_all_datasets.py.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="TS-GNN Optuna hyperparameter search")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Base config YAML (default: configs/default.yaml)")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of Optuna trials")
    p.add_argument("--n-epochs", type=int, default=30,
                   help="Training epochs per trial (short run)")
    p.add_argument("--timeout", type=int, default=None,
                   help="Max wall-clock seconds for the search")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Override TSGNN_DATA_DIR")
    p.add_argument("--out-config", type=str, default="configs/optuna_best.yaml",
                   help="Where to save the best config YAML")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    p.add_argument("--study-name", type=str, default="tsgnn_optuna")
    p.add_argument("--device", type=str, default=None,
                   help="'cuda', 'cpu', or None for autodetect")
    return p.parse_args()


def load_data_for_search(config: dict):
    """
    Load real BRCA data and build train/val splits for hyperparameter search.
    Mirrors the data loading logic in run_pipeline.py.
    Returns (train_data, val_data, esm_embeddings) or raises if data unavailable.
    """
    from tsgnn.data.brca_loader import load_brca_data, build_brca_temporal_sequences
    from tsgnn.data.preprocess import preprocess_scrna, select_features
    from tsgnn.data.grn_construction import construct_base_grn
    from tsgnn.data.allele import generate_esm2_embeddings, load_esm2_embeddings

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    temporal_cfg = config.get("temporal", {})

    n_genes = data_cfg.get("n_genes", model_cfg.get("input_dim", 500))
    K = temporal_cfg.get("K", 10)

    logger.info("Loading BRCA data...")
    adata = load_brca_data(malignant_only=True)
    adata = preprocess_scrna(adata)
    gene_list = select_features(adata, n_genes=n_genes)

    logger.info("Building GRN...")
    edge_index, edge_weight, _ = construct_base_grn(
        gene_list, expr_matrix=adata[:, gene_list].X
    )

    logger.info("Building temporal sequences...")
    all_sequences = build_brca_temporal_sequences(
        adata, gene_list, edge_index, K=K
    )

    # Patient-level train/val split (70/30)
    alleles = list(all_sequences.keys())
    n_train = max(1, int(len(alleles) * 0.7))
    train_data = {a: all_sequences[a] for a in alleles[:n_train]}
    val_data = {a: all_sequences[a] for a in alleles[n_train:]}

    if not val_data:
        # If only 1 allele, use same data for val (for development)
        val_data = train_data

    # ESM-2 embeddings
    logger.info("Loading ESM-2 embeddings...")
    esm_embeddings = generate_esm2_embeddings()
    if not esm_embeddings:
        esm_embeddings = load_esm2_embeddings()
    if not esm_embeddings:
        seed_val = config.get("seed", 42)
        logger.warning(
            f"No ESM-2 embeddings found. Using reproducible random placeholders "
            f"(seed={seed_val}). Run download_all_datasets.py for real embeddings."
        )
        from tsgnn.data.allele import TP53_HOTSPOT_MUTATIONS
        _rng = torch.Generator()
        _rng.manual_seed(seed_val)
        esm_embeddings = {"WT": torch.randn(1280, generator=_rng)}
        for allele in TP53_HOTSPOT_MUTATIONS:
            esm_embeddings[allele] = torch.randn(1280, generator=_rng)

    return train_data, val_data, esm_embeddings


def main():
    args = parse_args()

    if args.data_dir:
        os.environ["TSGNN_DATA_DIR"] = args.data_dir

    # Load base config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    # Set global seed
    seed = base_config.get("seed", 42)
    torch.manual_seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    try:
        train_data, val_data, esm_embeddings = load_data_for_search(base_config)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        logger.error(
            "Ensure BRCA data is downloaded (run scripts/download_all_datasets.py) "
            "and TSGNN_DATA_DIR is set."
        )
        sys.exit(1)

    # Run Optuna search
    from tsgnn.training.optuna_search import run_optuna_search, print_study_summary

    results = run_optuna_search(
        train_data=train_data,
        val_data=val_data,
        esm_embeddings=esm_embeddings,
        edge_index=edge_index,
        base_config=base_config,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs,
        timeout=args.timeout,
        device=device,
        out_config_path=args.out_config,
        study_name=args.study_name,
        storage=args.storage,
    )

    print_study_summary(results["study"])
    logger.info(f"Best config saved to: {args.out_config}")


if __name__ == "__main__":
    main()
