"""
scRNA-seq preprocessing pipeline for TS-GNN.

Phase 1E.1-2: Quality control, normalization, batch correction,
              malignant cell identification, dimensionality reduction.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from tsgnn.data import DATA_DIR
from tsgnn.data.grn_construction import BRCA_KEY_TFS


def _assign_fallback_clusters(adata, key: str = "leiden", n_clusters: int = 10) -> None:
    """Create lightweight cluster labels when leiden backends are unavailable."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        adata.obs[key] = "0"
        logger.warning("sklearn not installed; using a single fallback cluster.")
        return

    if "X_pca" in adata.obsm:
        features = adata.obsm["X_pca"]
    elif "X_umap" in adata.obsm:
        features = adata.obsm["X_umap"]
    else:
        X = adata.X
        features = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    n_obs = features.shape[0]
    n_clusters = max(1, min(n_clusters, n_obs))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(features)
    adata.obs[key] = labels.astype(str)
    logger.warning(
        "Leiden backend unavailable; assigned fallback KMeans clusters to adata.obs['%s'].",
        key,
    )


def _apply_soupx_if_available(adata):
    """
    Apply SoupX ambient RNA correction if pre-computed counts are available.
    If the 'soupx_corrected' layer already exists (from external R preprocessing), use it.
    Otherwise log instructions for running SoupX in R.
    """
    if "soupx_corrected" in adata.layers:
        adata.X = adata.layers["soupx_corrected"]
        logger.info("SoupX: corrected counts loaded from adata.layers['soupx_corrected']")
    else:
        logger.info(
            "SoupX correction not applied. To apply it in R:\n"
            "  Rscript -e \"\n"
            "    library(SoupX)\n"
            "    sc = load10X('path/to/cellranger/output')\n"
            "    sc = autoEstCont(sc)\n"
            "    out = adjustCounts(sc)\n"
            "    saveRDS(out, 'soupx_corrected_counts.rds')\n"
            "  \"\n"
            "Then load as adata.layers['soupx_corrected'] before preprocessing."
        )


def _batch_correct_scvi(adata, batch_key: str) -> str:
    """
    Probabilistic batch correction via scVI (Lopez et al. 2018, Nature Methods).

    scVI learns a deep generative model of scRNA-seq data with a batch covariate,
    producing a batch-corrected latent representation (10-dim default). This is the
    gold-standard method for multi-patient integration, outperforming Harmony on
    datasets with >10 batches (Luecken et al. 2021 benchmark, Nature Methods).

    Requires: pip install scvi-tools

    Returns:
        Key in adata.obsm that stores the corrected representation ("X_scVI").
    """
    try:
        import scvi

        logger.info(
            f"Running scVI batch correction on '{batch_key}' "
            f"({adata.obs[batch_key].nunique()} batches)..."
        )

        # scVI requires raw counts in adata.X or adata.layers["counts"]
        if "counts" in adata.layers:
            import anndata as ad
            adata_raw = ad.AnnData(
                X=adata.layers["counts"],
                obs=adata.obs,
                var=adata.var,
            )
        else:
            logger.warning(
                "scVI: raw counts layer not found; using current adata.X. "
                "Results may be suboptimal if data is already log-normalised."
            )
            adata_raw = adata.copy()

        scvi.model.SCVI.setup_anndata(adata_raw, batch_key=batch_key)
        model = scvi.model.SCVI(
            adata_raw,
            n_latent=10,
            n_layers=2,
            n_hidden=128,
            gene_likelihood="nb",   # negative binomial for scRNA
        )
        model.train(
            max_epochs=100,
            early_stopping=True,
            early_stopping_patience=10,
            plan_kwargs={"lr": 1e-3},
            progress_bar_refresh_rate=0,  # suppress lightning progress bar
        )

        adata.obsm["X_scVI"] = model.get_latent_representation()
        logger.info(
            f"scVI correction complete: latent shape {adata.obsm['X_scVI'].shape}"
        )
        return "X_scVI"

    except ImportError:
        logger.warning(
            "scvi-tools not installed. Falling back to Harmony. "
            "Install with: pip install scvi-tools"
        )
        try:
            import harmonypy
            ho = harmonypy.run_harmony(
                adata.obsm["X_pca"], adata.obs, batch_key, max_iter_harmony=20
            )
            Z = ho.Z_corr
            adata.obsm["X_pca_harmony"] = Z if Z.shape[0] == adata.n_obs else Z.T
            logger.info("Harmony fallback applied.")
            return "X_pca_harmony"
        except ImportError:
            logger.warning("harmonypy also not installed. No batch correction applied.")
            return "X_pca"
    except Exception as e:
        logger.warning(f"scVI training failed ({e}). Falling back to Harmony.")
        try:
            import harmonypy
            ho = harmonypy.run_harmony(
                adata.obsm["X_pca"], adata.obs, batch_key, max_iter_harmony=20
            )
            Z = ho.Z_corr
            adata.obsm["X_pca_harmony"] = Z if Z.shape[0] == adata.n_obs else Z.T
            return "X_pca_harmony"
        except Exception:
            return "X_pca"


def preprocess_scrna(
    adata,
    n_top_genes: int = 2000,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    batch_key: Optional[str] = None,
    n_pcs: int = 50,
    batch_method: str = "harmony",
    causal_scaling: bool = False,
    pseudotime_key: Optional[str] = "pseudotime",
):
    """
    Full scRNA-seq preprocessing pipeline.

    Steps:
    1. Basic QC filtering
    2. Doublet removal (Scrublet)
    3. Normalization (scanpy normalize_total + log1p)
    4. HVG selection
    5. Batch correction (Harmony, if batch_key provided)
    6. PCA + UMAP

    Args:
        adata: Raw AnnData object with counts.
        n_top_genes: Number of highly variable genes to select.
        min_genes: Minimum genes per cell.
        min_cells: Minimum cells per gene.
        max_pct_mito: Maximum mitochondrial gene percentage.
        batch_key: Column in adata.obs for batch correction.
        n_pcs: Number of PCA components.
        causal_scaling: If True, scale features using only the earliest timepoint statistics.
        pseudotime_key: Column in adata.obs identifying temporal progression.

    Returns:
        Preprocessed AnnData object.
    """
    import scanpy as sc

    logger.info(f"Starting preprocessing: {adata.n_obs} cells, {adata.n_vars} genes")

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    # ── Step 0: SoupX ──
    _apply_soupx_if_available(adata)

    # ── Step 1: Basic QC ──
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Mitochondrial QC
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
    logger.info(f"QC filtering: {n_before} -> {adata.n_obs} cells")

    # ── Step 2: Doublet removal ──
    try:
        import scrublet as scr
        scrub = scr.Scrublet(adata.layers["counts"])
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
        adata.obs["doublet_score"] = doublet_scores
        adata.obs["is_doublet"] = predicted_doublets
        n_doublets = predicted_doublets.sum()
        adata = adata[~adata.obs["is_doublet"]].copy()
        logger.info(f"Removed {n_doublets} doublets, {adata.n_obs} cells remaining")
    except ImportError:
        logger.warning("Scrublet not installed. Skipping doublet removal.")

    # ── Step 3: Normalization ──
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()  # Store normalized data before HVG filtering

    # ── Step 4: HVG selection ──
    # seurat_v3 requires raw counts; fall back to seurat flavour if unavailable
    if "counts" in adata.layers:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                    flavor="seurat_v3", layer="counts")
    else:
        logger.warning(
            "Raw counts layer ('counts') not found; falling back to 'seurat' HVG flavour. "
            "Store raw counts in adata.layers['counts'] before calling preprocess_scrna() "
            "for best results."
        )
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    n_hvgs = adata.var["highly_variable"].sum()
    logger.info(f"Selected {n_hvgs} highly variable genes")

    # ── Step 5: Subset to HVGs for dimensionality reduction ──
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    
    if causal_scaling and pseudotime_key in adata.obs:
        # Causal Scaling: compute mean/std only on the earliest timepoint (bin 0)
        # to prevent "future state bias" where inductions at t=10 affect
        # the normalized magnitude of features at t=0.
        # Proving myself wrong: Standard sc.pp.scale is transductive; 
        # Causal scaling is inductive, mimicking a real-time production inference.
        logger.info(f"Applying CAUSAL SCALING using {pseudotime_key}...")
        
        # Identify "early" cells (those in the minimum pseudotime bin/value)
        min_p = adata.obs[pseudotime_key].min()
        early_mask = adata.obs[pseudotime_key] <= min_p
        
        if early_mask.sum() > 5: # Need enough cells for stable variance
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            
            # Use raw counts if available for better variance estimation,
            # but standard practice is log-normalized for scaling.
            X_all = adata_hvg.X
            if hasattr(X_all, "toarray"): X_all = X_all.toarray()
            
            if np.isnan(X_all).any():
                logger.warning("X_all contains NaNs BEFORE causal scaling!")
            
            # Fit ONLY on early cells
            # We use a robust approach: if variance is 0, we don't scale that gene
            # to avoid NaNs, matching sc.pp.scale behavior.
            early_stats = X_all[early_mask.values]
            means = early_stats.mean(axis=0)
            stds = early_stats.std(axis=0)
            
            # Prevent division by zero
            stds[stds == 0] = 1.0
            
            # Manual scaling to ensure robustness
            X_scaled = (X_all - means) / stds
            adata_hvg.X = np.clip(X_scaled, None, 10)
        else:
            logger.warning(
                f"Causal scaling failed: only {early_mask.sum()} early cells found. "
                "Falling back to global scaling."
            )
            sc.pp.scale(adata_hvg, max_value=10)
    else:
        sc.pp.scale(adata_hvg, max_value=10)

    # ── Step 6: PCA ──
    sc.tl.pca(adata_hvg, n_comps=n_pcs)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    # ── Step 7: Batch correction ──
    pca_key = "X_pca"
    if batch_key and batch_key in adata.obs.columns:
        if batch_method == "scvi":
            pca_key = _batch_correct_scvi(adata, batch_key)
        else:
            # Default: Harmony (fast, PCA-space correction)
            try:
                import harmonypy
                ho = harmonypy.run_harmony(
                    adata.obsm["X_pca"], adata.obs, batch_key, max_iter_harmony=20
                )
                # Z_corr shape differs across harmonypy versions:
                # older: (n_PCs, n_cells) → need .T
                # newer: (n_cells, n_PCs) → use as-is
                Z = ho.Z_corr
                adata.obsm["X_pca_harmony"] = Z if Z.shape[0] == adata.n_obs else Z.T
                logger.info(f"Harmony batch correction applied on '{batch_key}'")
                pca_key = "X_pca_harmony"
            except ImportError:
                logger.warning("harmonypy not installed. Skipping batch correction.")
    elif batch_method == "scvi":
        logger.warning("batch_method='scvi' requires batch_key; skipping scVI correction.")

    # ── Step 8: Neighbors + UMAP ──
    sc.pp.neighbors(adata, use_rep=pca_key, n_pcs=n_pcs)
    sc.tl.umap(adata)
    try:
        # scanpy >= 1.10: use igraph backend (faster, required for future versions)
        sc.tl.leiden(adata, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
    except (TypeError, ImportError):
        try:
            # scanpy < 1.10: flavor parameter not available
            # optional dependency missing: igraph/leidenalg unavailable
            sc.tl.leiden(adata, resolution=1.0)
        except ImportError:
            _assign_fallback_clusters(adata)

    logger.info(f"Preprocessing complete: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def identify_malignant_cells(
    adata,
    method: str = "copykat",
    reference_groups: Optional[list] = None,
) -> None:
    """
    Identify malignant cells via copy number inference.

    Adds adata.obs["is_malignant"] (bool) column.

    Args:
        adata: Preprocessed AnnData.
        method: "inferCNV" or "copykat" (copykat is faster).
        reference_groups: Non-malignant cell types to use as reference.
    """
    logger.info(f"Identifying malignant cells using {method}...")

    if method == "copykat":
        logger.info(
            "CopyKAT requires R. For now, marking all cells as potentially malignant.\n"
            "To run CopyKAT:\n"
            "  1. Export counts: adata.layers['counts'] to CSV\n"
            "  2. Run CopyKAT in R\n"
            "  3. Import predictions back\n"
            "Alternatively, use cell type annotations to exclude known non-malignant types."
        )
        # Heuristic: if cell type annotations exist, use them
        if "cell_type" in adata.obs.columns:
            non_malignant = {"T cell", "B cell", "Macrophage", "Fibroblast",
                             "Endothelial", "Mast cell", "NK cell", "Plasma"}
            adata.obs["is_malignant"] = ~adata.obs["cell_type"].isin(non_malignant)
            n_mal = adata.obs["is_malignant"].sum()
            logger.info(f"Heuristic malignant identification: {n_mal}/{adata.n_obs} cells")
        else:
            adata.obs["is_malignant"] = True
            logger.warning("No cell type annotations. Marking all cells as malignant.")

    elif method == "copykat_rpy2":
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()

            counts_df = pd.DataFrame.sparse.from_spmatrix(
                adata.layers["counts"],
                index=adata.obs_names,
                columns=adata.var_names,
            ).T  # CopyKAT vuole geni × cellule

            ro.globalenv["counts_r"] = pandas2ri.py2rpy(counts_df)
            ro.r("""
                library(copykat)
                copykat_result <- copykat(
                    rawmat     = as.matrix(counts_r),
                    id.type    = "S",
                    ngene.chr  = 5,
                    win.size   = 25,
                    KS.cut     = 0.1,
                    sam.name   = "tsgnn",
                    n.cores    = 4
                )
                pred <- copykat_result$prediction
            """)
            pred = pandas2ri.rpy2py(ro.globalenv["pred"])
            adata.obs["is_malignant"] = (pred["copykat.pred"] == "aneuploid").values
            logger.info(f"CopyKAT: {adata.obs['is_malignant'].sum()} malignant cells identified")

        except ImportError:
            logger.warning("rpy2 not installed — fallback: marking every cell as malignant.")
            adata.obs["is_malignant"] = True
        except Exception as e:
            logger.warning(f"CopyKAT rpy2 failed: {e} — fallback: marking every cell as malignant.")
            adata.obs["is_malignant"] = True

    elif method == "infercnv":
        logger.info(
            "inferCNV requires R and significant runtime (2-4h).\n"
            "Export adata and run externally, then import results."
        )
        adata.obs["is_malignant"] = True

    n_malignant = adata.obs["is_malignant"].sum()
    logger.info(f"Malignant cells identified: {n_malignant}/{adata.n_obs}")


def select_features(
    adata,
    n_genes: int = 500,
    tf_list: Optional[list] = None,
    p53_targets: Optional[pd.DataFrame] = None,
) -> list:
    """
    Select features for GRN modeling.

    Priority:
    1. All TFs from JASPAR 2024 present in the data
    2. Known p53 target genes from TargetGeneReg 2.0
    3. Top HVGs to fill remaining slots

    Args:
        adata: Preprocessed AnnData.
        n_genes: Target number of genes (N).
        tf_list: List of TF gene symbols from JASPAR.
        p53_targets: DataFrame with p53 target gene names.

    Returns:
        List of selected gene names.
    """
    # BRCA-specific key TFs — always prioritised before generic JASPAR list
    # (canonical list defined in grn_construction.BRCA_KEY_TFS, imported above)
    all_genes = set(adata.var_names)
    selected = set()

    # Priority 0: BRCA key TFs
    brca_tfs_in_data = BRCA_KEY_TFS & all_genes
    selected.update(brca_tfs_in_data)
    if brca_tfs_in_data:
        logger.info(f"BRCA key TFs in data: {sorted(brca_tfs_in_data)}")

    # Priority 1: TFs present in data
    if tf_list:
        tfs_in_data = set(tf_list) & all_genes
        selected.update(tfs_in_data)
        logger.info(f"TFs in data: {len(tfs_in_data)}/{len(tf_list)}")

    # Priority 2: p53 targets present in data
    if p53_targets is not None:
        target_genes = set(p53_targets["gene_name"]) & all_genes
        selected.update(target_genes)
        logger.info(f"p53 targets in data: {len(target_genes)}")

    # Always include TP53 itself
    if "TP53" in all_genes:
        selected.add("TP53")

    # Priority 3: Fill with HVGs
    if len(selected) < n_genes and "highly_variable" in adata.var.columns:
        hvgs = adata.var_names[adata.var["highly_variable"]].tolist()
        for g in hvgs:
            if len(selected) >= n_genes:
                break
            selected.add(g)

    # If still not enough, add by variance
    if len(selected) < n_genes:
        import scanpy as sc
        if hasattr(adata.var, "dispersions_norm"):
            remaining = sorted(
                all_genes - selected,
                key=lambda g: adata.var.loc[g, "dispersions_norm"]
                if g in adata.var.index else 0,
                reverse=True,
            )
            for g in remaining:
                if len(selected) >= n_genes:
                    break
                selected.add(g)

    selected = sorted(selected)[:n_genes]
    logger.info(f"Feature selection: {len(selected)} genes selected (target: {n_genes})")
    return selected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Preprocessing module ready. Import and use with AnnData objects.")
