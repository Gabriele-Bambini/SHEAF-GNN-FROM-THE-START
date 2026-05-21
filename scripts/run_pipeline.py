"""
End-to-End TS-GNN Pipeline.

Phase 6: Executes the complete pipeline from raw data to final results.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --config configs/phase0_crc.yaml --skip-download
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger("tsgnn.pipeline")


def _strict_science_enabled(config: dict) -> bool:
    env = os.environ.get("TSGNN_STRICT_SCIENCE")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    return bool(config.get("science", {}).get("strict_real_data", True))


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_pipeline(config_path: str, skip_download: bool = False, skip_training: bool = False,
                 preprocess_only: bool = False):
    """Execute the full TS-GNN pipeline."""
    start = time.time()

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, Seed: {seed}")
    strict_science = _strict_science_enabled(config)

    from tsgnn.data import DATA_DIR
    data_dir = Path(config.get("data_dir") or os.environ.get("TSGNN_DATA_DIR") or DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # PHASE 1: DATA ACQUISITION
    # =====================================================================
    # FORCE DELETE ALL PREVIOUS CHECKPOINTS TO PREVENT POISONED RESUMPTION
    import shutil
    try:
        colab_ckpt_dir = Path(config_path).resolve().parent.parent / "checkpoints"
        if colab_ckpt_dir.exists():
            shutil.rmtree(colab_ckpt_dir)
            logger.info(f"HARD DELETED checkpoints directory at {colab_ckpt_dir} to ensure clean run.")
    except Exception as e:
        logger.warning(f"Could not hard delete checkpoints: {e}")

    logger.info("=" * 60)
    logger.info("PHASE 1: DATA ACQUISITION AND PREPROCESSING")
    logger.info("=" * 60)

    if not skip_download:
        try:
            from tsgnn.data.download import download_all
            download_all(str(data_dir))
        except Exception as e:
            logger.warning(f"Download failed (may need manual download): {e}")

    # Generate ESM-2 embeddings (uses module's built-in ESM_DIR path)
    from tsgnn.data.allele import (
        TP53_HOTSPOT_MUTATIONS,
        load_esm2_embeddings,
        generate_esm2_embeddings,
    )

    logger.info("Generating/loading ESM-2 embeddings...")
    required_alleles = list(dict.fromkeys(config.get("data", {}).get("alleles", list(TP53_HOTSPOT_MUTATIONS.keys()))))
    required_embedding_names = list(dict.fromkeys(["WT", *required_alleles]))
    esm_embeddings = load_esm2_embeddings(required_alleles)
    missing_embeddings = [a for a in required_embedding_names if a not in esm_embeddings]
    if missing_embeddings:
        try:
            generated = generate_esm2_embeddings(
                alleles=[a for a in required_alleles if a in missing_embeddings],
                allow_placeholder=not strict_science,
            )
            esm_embeddings.update(generated)
        except Exception as e:
            if strict_science:
                raise RuntimeError(
                    "Strict scientific mode requires real ESM-2 embeddings. "
                    f"Missing alleles: {missing_embeddings}."
                ) from e
            logger.warning(f"ESM-2 generation failed: {e}. Using placeholders.")

    # Fallback: generate random embeddings in memory
    if not esm_embeddings or any(a not in esm_embeddings for a in required_embedding_names):
        if strict_science:
            missing_after = [a for a in required_embedding_names if a not in esm_embeddings]
            raise FileNotFoundError(
                "Strict scientific mode forbids placeholder ESM embeddings. "
                f"Still missing: {missing_after}"
            )
        seed_val = config.get("seed", 42)
        logger.warning(
            f"No ESM-2 embeddings available. Generating reproducible random placeholders "
            f"(seed={seed_val}). Results will NOT be biologically meaningful — "
            "run `python scripts/download_all_datasets.py` to compute real ESM-2 embeddings."
        )
        from tsgnn.data.allele import TP53_HOTSPOT_MUTATIONS
        _rng = torch.Generator()
        _rng.manual_seed(seed_val)
        esm_embeddings = {"WT": torch.randn(1280, generator=_rng)}
        for allele in TP53_HOTSPOT_MUTATIONS:
            esm_embeddings[allele] = torch.randn(1280, generator=_rng)

    logger.info(f"Loaded ESM-2 embeddings for {len(esm_embeddings)} alleles")

    # =====================================================================
    # PHASE 1E: DATA INTEGRATION — REAL DATA ONLY
    # No synthetic fallback. If real data is missing, the pipeline fails
    # with a clear error message.
    # =====================================================================
    model_cfg = config.get("model", {})
    data_cfg  = config.get("data", {})
    temporal_cfg = config.get("temporal", {})
    K  = data_cfg.get("K", temporal_cfg.get("K", 10))
    n_genes = data_cfg.get("n_genes", 500)

    from tsgnn.data.brca_loader import load_brca_data, build_brca_temporal_sequences
    from tsgnn.data.preprocess import preprocess_scrna, select_features
    from tsgnn.data.grn_construction import (
        load_regnetwork, load_targetgenereg,
        load_string_ppi, construct_base_grn,
    )

    # ── Preprocessing cache: save/load processed AnnData to skip slow re-processing ──
    import scanpy as sc
    processed_dir = Path(data_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    adata_cache = processed_dir / "brca_preprocessed.h5ad"

    ext_dir = Path(data_dir) / "external"
    jaspar_file = ext_dir / "jaspar" / "JASPAR2024_CORE_vertebrates.txt"
    tf_list = None
    if jaspar_file.exists():
        with open(jaspar_file) as f:
            tf_list = [line.strip().lstrip(">") for line in f if line.startswith(">")]

    targetgenereg_df = load_targetgenereg()

    malignant_raw_cache = processed_dir / "brca_malignant_raw.h5ad"

    if adata_cache.exists():
        logger.info(f"Loading preprocessed AnnData from cache: {adata_cache}")
        adata = sc.read_h5ad(adata_cache)
        logger.info(f"Cache loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    else:
        # Try intermediate raw-malignant cache before loading from MTX
        if malignant_raw_cache.exists():
            logger.info(f"Loading malignant raw cache (skipping MTX read): {malignant_raw_cache}")
            adata = sc.read_h5ad(malignant_raw_cache)
            logger.info(f"Raw cache loaded: {adata.n_obs} cells, {adata.n_vars} genes")
        else:
            logger.info("Loading real BRCA scRNA-seq data (no synthetic fallback)...")
            adata = load_brca_data(use_gse158508=False, malignant_only=True)
            adata.write_h5ad(malignant_raw_cache)
            logger.info(f"Malignant raw AnnData saved to cache: {malignant_raw_cache}")

        adata = preprocess_scrna(
            adata,
            n_top_genes=data_cfg.get("n_hvgs", 2000),
            batch_key="patient_id",
        )
        adata.write_h5ad(adata_cache)
        logger.info(f"Preprocessed AnnData saved to cache: {adata_cache}")

    grn_cache = processed_dir / "grn_artifacts.pt"
    seq_cache = processed_dir / "temporal_sequences.pt"

    if grn_cache.exists():
        logger.info(f"Loading GRN from cache: {grn_cache}")
        grn_data = torch.load(grn_cache, map_location="cpu", weights_only=False)
        gene_list  = grn_data["gene_list"]
        edge_index = grn_data["edge_index"]
        edge_weight= grn_data["edge_weight"]
        N = grn_data["N"]
        E = grn_data["E"]
        logger.info(f"GRN cache loaded: {N} nodes, {E} edges")
    else:
        gene_list = select_features(
            adata, n_genes=n_genes, tf_list=tf_list, p53_targets=targetgenereg_df
        )
        N = len(gene_list)
        logger.info(f"Feature selection: {N} genes")

        # Build GRN from regulatory priors
        regnet_df = load_regnetwork()
        _str_dir = ext_dir / "string"
        _str_cands = sorted(_str_dir.glob("*protein.links*")) if _str_dir.exists() else []
        _str_path = _str_cands[0] if _str_cands else None
        if _str_path:
            logger.info(f"STRING file: {_str_path.name}")
        string_df = load_string_ppi(_str_path) if _str_path else None

        import numpy as np
        expr_matrix = None
        if adata.raw is not None:
            gene_indices = [
                list(adata.raw.var_names).index(g)
                for g in gene_list if g in adata.raw.var_names
            ]
            mat = adata.raw.X[:, gene_indices]
            if hasattr(mat, "toarray"):
                mat = mat.toarray()
            expr_matrix = np.asarray(mat, dtype=np.float32)

        scprint_edges = None
        if data_cfg.get("use_scprint_grn", False):
            try:
                from tsgnn.data.scprint_grn import get_scprint_grn_edges
                scprint_edges = get_scprint_grn_edges(
                    adata=adata,
                    gene_names=gene_list,
                    k=8,  # Default or pull from config
                    max_cells=25000,
                    cache_dir=Path(processed_dir),
                )
                logger.info(f"Loaded scPRINT edges: {scprint_edges.shape[1]}")
            except Exception as e:
                logger.warning(f"scPRINT inference failed: {e}. Continuing without scPRINT.")

        edge_index, edge_weight, grn_meta = construct_base_grn(
            gene_list=gene_list,
            regnetwork_edges=regnet_df,
            targetgenereg=targetgenereg_df,
            string_ppi=string_df,
            expr_matrix=expr_matrix,
            scprint_edges=scprint_edges,
        )
        E = edge_index.shape[1]
        logger.info(f"Real GRN: {N} nodes, {E} edges")

        torch.save({
            "gene_list":   gene_list,
            "edge_index":  edge_index,
            "edge_weight": edge_weight,
            "N": N, "E": E,
        }, grn_cache)
        logger.info(f"GRN artifacts saved to cache: {grn_cache}")

    logger.info(f"Working graph: {N} nodes, {E} edges")

    if seq_cache.exists():
        logger.info(f"Loading temporal sequences from cache: {seq_cache}")
        seq_data   = torch.load(seq_cache, map_location="cpu", weights_only=False)
        train_data = seq_data["train_data"]
        val_data   = seq_data["val_data"]
        alleles    = seq_data["alleles"]
        real_esm   = seq_data["real_esm"]
        logger.info(f"Sequence cache loaded: {len(alleles)} alleles: {alleles}")
    else:
        # ── Build per-allele temporal sequences (real data) ───────────────
        sequences = build_brca_temporal_sequences(
            adata=adata,
            gene_list=gene_list,
            edge_index=edge_index,
            K=K,
            binning_method=data_cfg.get("binning_method", temporal_cfg.get("binning_method", "adaptive")),
            min_cells_per_allele=data_cfg.get("min_cells_per_allele", 50),
        )
        alleles = list(sequences.keys())
        if not alleles:
            raise RuntimeError(
                "No allele has enough cells to build temporal sequences. "
                "Check wu2021_tp53_alleles.csv and min_cells_per_allele config."
            )
        logger.info(f"Built temporal sequences for {len(alleles)} alleles: {alleles}")

        train_data, val_data = {}, {}
        for allele, seq in sequences.items():
            nf = seq.node_features_sequence   # (K, N, d)
            ew = seq.edge_weights_sequence    # (K, E)
            split = max(1, int(0.85 * K))
            nf_train = nf[:split]
            nf_val   = nf[split:]
            # Targets = NEXT time step (t+1), not current (t).
            # Identity targets (t→t) produce a trivial autoencoder with near-zero
            # loss but zero allele specificity. Next-step targets force the model
            # to learn meaningful temporal dynamics and allele-conditioned rewiring.
            # At the last step we repeat the final frame (no future available).
            K_train = nf_train.shape[0]
            K_val   = nf_val.shape[0]
            train_data[allele] = {
                "node_features_seq": nf_train,
                "edge_weights_seq":  ew[:split],
                "targets": [nf_train[min(k + 1, K_train - 1)] for k in range(K_train)],
                "bin_boundaries": seq.bin_boundaries,
            }
            val_data[allele] = {
                "node_features_seq": nf_val,
                "edge_weights_seq":  ew[split:],
                "targets": [nf_val[min(k + 1, K_val - 1)] for k in range(K_val)],
                "bin_boundaries": seq.bin_boundaries,
            }

        # Only use ESM2 embeddings for alleles that have real data
        missing_real_esm = [a for a in alleles if a not in esm_embeddings]
        if missing_real_esm and strict_science:
            raise FileNotFoundError(
                "Strict scientific mode forbids WT fallback for missing allele embeddings. "
                f"Missing: {missing_real_esm}"
            )
        real_esm = {a: esm_embeddings.get(a, esm_embeddings["WT"]) for a in alleles}
        if missing_real_esm:
            logger.warning(f"Missing ESM2 embeddings for {missing_real_esm}, using WT as fallback")

        torch.save({
            "train_data": train_data,
            "val_data":   val_data,
            "alleles":    alleles,
            "real_esm":   real_esm,
            "K": K, "N": N, "E": E,
        }, seq_cache)
        logger.info(f"Temporal sequences saved to cache: {seq_cache}")

    # ── Preprocess-only mode: stop here, all artifacts are cached ────────
    if preprocess_only:
        logger.info("=" * 60)
        logger.info("PREPROCESS-ONLY mode: all artifacts saved. Stopping before training.")
        logger.info(f"  brca_preprocessed.h5ad : {adata_cache}")
        logger.info(f"  grn_artifacts.pt       : {grn_cache}")
        logger.info(f"  temporal_sequences.pt  : {seq_cache}")
        logger.info("Upload the data/processed/ folder to Google Drive, then run Colab.")
        logger.info("=" * 60)
        elapsed = time.time() - start
        logger.info(f"Done in {elapsed:.0f}s")
        return

    # =====================================================================
    # PHASE 2: MODEL CONSTRUCTION
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL CONSTRUCTION")
    logger.info("=" * 60)

    from tsgnn.model.tsgnn import TSGNN

    d = model_cfg.get("stalk_dim", 4)
    # Derive input_dim from actual data shape (features per node per time step)
    _sample_allele = alleles[0]
    _sample_seq = train_data[_sample_allele]["node_features_seq"]  # (K, N, feat_dim)
    actual_input_dim = _sample_seq.shape[-1]
    input_dim = model_cfg.get("input_dim", actual_input_dim)
    if input_dim != actual_input_dim:
        logger.warning(
            f"Config input_dim={input_dim} != data feature dim={actual_input_dim}. "
            f"Using actual data dim={actual_input_dim}."
        )
        input_dim = actual_input_dim
    logger.info(f"Model input_dim={input_dim} (features per node), N={N} nodes")
    model = TSGNN(
        num_nodes=N,
        num_edges=E,
        stalk_dim=d,
        input_dim=input_dim,
        esm_dim=model_cfg.get("esm_dim", 1280),
        conditioning_dim=model_cfg.get("conditioning_dim", 128),
        edge_index=edge_index,
        num_diffusion_steps=model_cfg.get("num_diffusion_steps", 3),
        use_allele_conditioning=True,
    )

    params = model.count_parameters()
    logger.info(f"Model parameters: {params['total']:,}")
    for name, count in params.items():
        if name != "total":
            logger.info(f"  {name}: {count:,}")

    # =====================================================================
    # PHASE 3: TRAINING
    # =====================================================================
    if not skip_training:
        logger.info("=" * 60)
        logger.info("PHASE 3: TRAINING")
        logger.info("=" * 60)

        from tsgnn.training.trainer import TSGNNTrainer

        trainer = TSGNNTrainer(
            model=model,
            config=config,
            device=device,
            checkpoint_dir=str(project_root / "checkpoints"),
            use_wandb=config.get("training", {}).get("use_wandb", False),
        )

        results = trainer.train(
            train_data=train_data,
            val_data=val_data,
            esm_embeddings=real_esm,   # real ESM-2 where available, random fallback
        )

        logger.info(f"Training complete: {results['epochs_trained']} epochs, "
                     f"best val loss: {results['best_val_loss']:.6f}")

        # ── Loss curve ────────────────────────────────────────────────────
        trainer.plot_loss_curve(
            output_dir=str(project_root / "figures"),
            show=True,          # renders inline in Colab
            filename="loss_curve.png",
        )
    else:
        logger.info("Skipping training (--skip-training flag)")

    # =====================================================================
    # PHASE 4: EVALUATION
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: EVALUATION")
    logger.info("=" * 60)

    from tsgnn.evaluation.metrics import evaluate_all

    eval_results = evaluate_all(
        model=model,
        test_data=val_data,
        esm_embeddings=real_esm,
        stalk_dim=d,
    )

    logger.info("Evaluation results:")
    for metric, values in eval_results.items():
        logger.info(f"  {metric}: {values}")

    # =====================================================================
    # PHASE 5: VISUALIZATION
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: VISUALIZATION")
    logger.info("=" * 60)

    fig_dir = str(project_root / "figures")
    try:
        # Get model outputs for visualization
        model.eval()
        with torch.no_grad():
            allele = alleles[0]
            node_seq = val_data[allele]["node_features_seq"].to(device)
            allele_emb = real_esm[allele].to(device)
            preds, maps_traj, laps = model(node_seq, allele_emb)

        from tsgnn.visualization.rewiring import identify_bottleneck_timepoints
        bottlenecks = identify_bottleneck_timepoints(
            [l.cpu() for l in laps], edge_index, d
        )
        logger.info(f"Found {len(bottlenecks)} bottleneck transitions")

        from tsgnn.visualization.attention import analyze_regulatory_mode
        modes = analyze_regulatory_mode(
            [m.cpu() for m in maps_traj], edge_index
        )
        mode_counts = {}
        for v in modes.values():
            mode_counts[v["mode"]] = mode_counts.get(v["mode"], 0) + 1
        logger.info(f"Edge regulatory modes: {mode_counts}")

        # Try to generate plots (requires matplotlib)
        try:
            from tsgnn.visualization.rewiring import plot_rewiring_trajectory
            plot_rewiring_trajectory(
                [l.cpu() for l in laps], edge_index, d,
                allele=allele, output_dir=fig_dir,
            )

            from tsgnn.visualization.networks import plot_allele_embedding_space
            plot_allele_embedding_space(real_esm, output_dir=fig_dir)

            logger.info(f"Figures saved to {fig_dir}")
        except ImportError as e:
            logger.warning(f"Plotting skipped (missing dependency): {e}")

    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TS-GNN End-to-End Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download step")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (use existing checkpoint)")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Stop after preprocessing+GRN+sequences, before training. "
                             "Use to prepare artifacts for Colab training.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    run_pipeline(str(config_path), args.skip_download, args.skip_training,
                 preprocess_only=args.preprocess_only)


if __name__ == "__main__":
    main()
