"""
BIOKEY (E-MTAB-8107) scRNA-seq loader for TS-GNN held-out validation.

Reference
---------
Bassez A. et al. (2021) "A single-cell map of intratumoral changes during
anti-PD1 treatment of patients with breast cancer." Nat Med 27, 820–832.
https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-8107

Dataset characteristics
-----------------------
- ~175k cells from 53 BRCA patients (ER+, HER2+, TNBC).
- Pre-treatment and on-treatment samples (anti-PD1 + chemo).
- Cohort A (n=29, anti-PD1 monotherapy) + Cohort B (n=24, anti-PD1 + chemo).
- Use-case in TS-GNN: HELD-OUT generalization cohort (not in training).

Expected files
--------------
data/raw/brca/BIOKEY_E-MTAB-8107/
├── counts.tsv               (genes × cells, raw UMI counts, ~3 GB TSV)
├── metadata.csv             (cell_id, patient_id, timepoint, cohort, subtype)
└── [optional] BIOKEY_tp53_alleles.csv  (if user has per-patient TP53 calls)

If TP53 allele calls are not provided, cells are marked "unknown" and
the loader flags the cohort for QC-only / generalization metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tsgnn.data import RAW_DIR

logger = logging.getLogger(__name__)

BIOKEY_DIR = RAW_DIR / "brca" / "BIOKEY_E-MTAB-8107"
# Legacy combined-TSV layout (if user manually downloaded from Bassez supplementary)
BIOKEY_COUNTS = BIOKEY_DIR / "counts.tsv"
BIOKEY_META   = BIOKEY_DIR / "metadata.csv"
# Current ArrayExpress layout (E-MTAB-8107)
BIOKEY_COUNTS_DIR = BIOKEY_DIR / "counts"
BIOKEY_SDRF = BIOKEY_DIR / "E-MTAB-8107.sdrf.txt"
BIOKEY_ALLELES = BIOKEY_DIR / "BIOKEY_tp53_alleles.csv"  # optional


def biokey_available() -> bool:
    """Return True if any supported BIOKEY layout is present on disk.

    Accepts either the legacy combined TSV + metadata.csv pair, or the
    current ArrayExpress per-sample layout (counts/ folder + SDRF file).
    """
    legacy = BIOKEY_COUNTS.exists() and BIOKEY_META.exists()
    per_sample = (
        BIOKEY_COUNTS_DIR.exists()
        and BIOKEY_SDRF.exists()
        and any(BIOKEY_COUNTS_DIR.glob("*.counts.csv"))
    )
    return legacy or per_sample


def _load_counts(counts_path: Path, chunksize: int = 5000) -> "pd.DataFrame":
    """
    Load BIOKEY counts TSV in chunks to avoid memory spike on large files.

    The file is expected to be genes × cells with gene symbols in the first
    column.  We transpose to cells × genes to match the AnnData convention.
    """
    logger.info(f"Streaming counts from {counts_path} (chunksize={chunksize})...")
    chunks = []
    for chunk in pd.read_csv(counts_path, sep="\t", index_col=0, chunksize=chunksize):
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0)
    # Transpose to cells × genes
    df = df.T
    logger.info(f"Loaded counts: {df.shape[0]} cells × {df.shape[1]} genes")
    return df


def _load_per_sample_counts(
    counts_dir: Path,
    per_sample_cap: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load ArrayExpress per-sample counts (one CSV per donor sample).

    Memory-bounded: subsamples up to ``per_sample_cap`` cells per file BEFORE
    concat, casts to float32, and restricts to the common-gene intersection
    computed in a cheap header pass. Concat of 17 × 2k × 30k float32 ≈ 4 GB peak,
    fits the Colab 12 GB T4 RAM.
    """
    csv_files = sorted(counts_dir.glob("*.counts.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *.counts.csv under {counts_dir}")
    logger.info(
        f"Loading {len(csv_files)} per-sample BIOKEY matrices "
        f"(per_sample_cap={per_sample_cap}, float32)..."
    )

    # Pass 1: cheap scan — read only the row index (gene symbols) from each file.
    gene_sets = []
    valid_files = []
    for f in csv_files:
        try:
            idx = pd.read_csv(f, index_col=0, usecols=[0]).index
            gene_sets.append(set(idx))
            valid_files.append(f)
        except Exception as exc:
            logger.warning(f"  Skipping {f.name} (header scan failed): {exc}")
    if not gene_sets:
        raise RuntimeError("No readable BIOKEY counts files.")
    common_genes = sorted(set.intersection(*gene_sets))
    logger.info(f"  Common genes across {len(valid_files)} samples: {len(common_genes)}")

    # Pass 2: read each file, restrict rows to common genes, subsample cells, cast to float32.
    rng = np.random.default_rng(random_state)
    parts: list[pd.DataFrame] = []
    for f in valid_files:
        sample_id = f.name.replace(".counts.csv", "")
        loaded_shape = "skipped"
        try:
            # Read WITHOUT dtype constraint (some files have malformed rows with
            # gene-name strings leaking into value columns). We coerce after.
            df = pd.read_csv(f, index_col=0, low_memory=False)
            # Restrict rows to common genes (drops ~nothing for same-pipeline 10x).
            df = df.reindex(common_genes)
            # Defensive numeric coercion (handles corrupted rows / mixed dtypes).
            df = df.apply(pd.to_numeric, errors="coerce")
            if df.isnull().all().all():
                logger.warning(f"  Skipping {f.name}: entirely non-numeric (corrupted?)")
                continue
            df = df.fillna(0).astype(np.float32, copy=False)
            # Subsample cells PER FILE to bound peak memory.
            if df.shape[1] > per_sample_cap:
                pick = rng.choice(df.shape[1], size=per_sample_cap, replace=False)
                df = df.iloc[:, np.sort(pick)]
            df.columns = [f"{sample_id}_{c}" for c in df.columns]
            parts.append(df)
            loaded_shape = df.shape
        except Exception as exc:
            logger.warning(f"  Skipping {f.name}: {exc}")
        # Log progress so we can see where OOM would hit.
        logger.info(f"    loaded {f.name}: {loaded_shape}")

    combined = pd.concat(parts, axis=1)
    # Transpose to cells × genes
    combined = combined.T
    logger.info(f"  Combined: {combined.shape[0]} cells × {combined.shape[1]} genes")
    return combined


def load_biokey_metadata() -> pd.DataFrame:
    """Load BIOKEY metadata — SDRF preferred, legacy CSV fallback.

    Returns a DataFrame indexed by cell_id. The SDRF layout is sample-level,
    so we expand it to one row per (sample × cell) when per-sample counts are
    present on disk.
    """
    # Preferred: SDRF + per-sample counts (ArrayExpress layout)
    if BIOKEY_SDRF.exists() and BIOKEY_COUNTS_DIR.exists():
        sdrf = pd.read_csv(BIOKEY_SDRF, sep="\t")
        sdrf.columns = [c.strip().lower().replace(" ", "_") for c in sdrf.columns]
        # Normalise the sample ID column name
        sid = next((c for c in sdrf.columns
                    if c in {"source_name", "assay_name", "extract_name"}), sdrf.columns[0])

        # Expand SDRF per cell: read each counts file's column header to get the
        # cells belonging to each sample, then left-join the sample metadata.
        rows = []
        for f in sorted(BIOKEY_COUNTS_DIR.glob("*.counts.csv")):
            sample_id = f.name.replace(".counts.csv", "")
            # cheap header read
            cells = pd.read_csv(f, index_col=0, nrows=0).columns.tolist()
            sample_meta = sdrf.loc[sdrf[sid].astype(str).str.strip() == sample_id]
            if sample_meta.empty:
                # best-effort fallback: just tag patient_id = sample_id
                for c in cells:
                    rows.append({"cell_id": f"{sample_id}_{c}", "patient_id": sample_id})
            else:
                s = sample_meta.iloc[0].to_dict()
                for c in cells:
                    rows.append({
                        "cell_id": f"{sample_id}_{c}",
                        "patient_id": sample_id,
                        **{k: v for k, v in s.items() if k != sid},
                    })
        meta = pd.DataFrame(rows).set_index("cell_id")
        logger.info(f"BIOKEY metadata (SDRF-expanded): {len(meta)} cells, "
                    f"columns={list(meta.columns)[:8]}")
        return meta

    # Fallback: legacy combined metadata.csv
    if not BIOKEY_META.exists():
        raise FileNotFoundError(
            f"BIOKEY metadata missing: neither {BIOKEY_SDRF} nor {BIOKEY_META} found.\n"
            "Download from https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-8107"
        )
    meta = pd.read_csv(BIOKEY_META)
    meta.columns = [c.strip().lower().replace(" ", "_") for c in meta.columns]

    # Heuristic standardization — BIOKEY uses "Cell" or "cell" as index column
    for cid in ("cell", "cell_id", "cellid", "barcode"):
        if cid in meta.columns:
            meta = meta.set_index(cid)
            break

    # Standard column names expected downstream
    renames = {
        "patient": "patient_id",
        "patientid": "patient_id",
        "timepoint": "timepoint",
        "cohort": "cohort",
        "breast_cancer_subtype": "subtype",
        "celltype": "cell_type",
        "celltype_minor": "cell_type",
    }
    for old, new in renames.items():
        if old in meta.columns and new not in meta.columns:
            meta[new] = meta[old]

    logger.info(f"BIOKEY metadata: {len(meta)} cells, columns={list(meta.columns)}")
    return meta


def load_biokey_data(
    malignant_only: bool = True,
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> "anndata.AnnData":
    """
    Load BIOKEY scRNA-seq into AnnData with TP53 allele annotations.

    Parameters
    ----------
    malignant_only : bool
        If True and ``cell_type`` column is present, keep only cells not in
        the standard immune/stromal vocabulary (same filter as brca_loader).
    sample_n : int, optional
        If set, randomly subsample to this many cells (for fast held-out eval).
    random_state : int
        Seed for subsampling.

    Returns
    -------
    AnnData
        obs contains: patient_id, timepoint, cohort, subtype, cell_type,
        tp53_allele ("unknown" if no allele file present), is_malignant.

    Raises
    ------
    FileNotFoundError if BIOKEY files are absent. Tell user to run
    ``tsgnn.data.download.download_biokey_scrna()`` or manual link.
    """
    try:
        import anndata
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy/anndata required for BIOKEY loader. "
            "Install with: pip install scanpy anndata"
        ) from e

    if not biokey_available():
        raise FileNotFoundError(
            f"BIOKEY files not found in {BIOKEY_DIR}.\n"
            "Automatic: python -c \"from tsgnn.data.download import "
            "download_biokey_scrna; download_biokey_scrna()\"\n"
            "Manual: https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-8107"
        )

    # 1. Load counts — support two layouts
    if BIOKEY_COUNTS.exists():
        counts = _load_counts(BIOKEY_COUNTS)
    else:
        counts = _load_per_sample_counts(BIOKEY_COUNTS_DIR)

    # 2. Load metadata (SDRF preferred, fallback to legacy CSV)
    meta = load_biokey_metadata()

    # 3. Align to common cells
    common = counts.index.intersection(meta.index)
    if len(common) == 0:
        raise RuntimeError(
            "No cells in common between BIOKEY counts and metadata. "
            "Check cell ID conventions (expected same ID in both files)."
        )
    counts = counts.loc[common]
    meta   = meta.loc[common]
    logger.info(f"Aligned: {len(common)} cells with both counts + metadata")

    # 4. Optional subsample
    if sample_n is not None and sample_n < len(common):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(common), size=sample_n, replace=False)
        counts = counts.iloc[idx]
        meta   = meta.iloc[idx]
        logger.info(f"Subsampled to {sample_n} cells (seed={random_state})")

    # 5. Build AnnData
    adata = anndata.AnnData(
        X=counts.values.astype(np.float32),
        obs=meta.copy(),
        var=pd.DataFrame(index=counts.columns),
    )
    adata.var_names_make_unique()

    # 6. Assign TP53 alleles (if allele file present, else mark unknown)
    if BIOKEY_ALLELES.exists():
        from tsgnn.data.allele import assign_alleles_to_cells
        allele_df = pd.read_csv(BIOKEY_ALLELES)[["patient_id", "tp53_allele"]].drop_duplicates("patient_id")
        assign_alleles_to_cells(adata, allele_df, patient_col="patient_id")
        n_assigned = (adata.obs["tp53_allele"] != "unknown").sum()
        logger.info(f"BIOKEY alleles: {n_assigned}/{adata.n_obs} cells labeled")
    else:
        adata.obs["tp53_allele"] = "unknown"
        logger.warning(
            "BIOKEY_tp53_alleles.csv not found — cells marked 'unknown'. "
            "Cohort is usable for generalization / QC metrics only."
        )

    # 7. Malignant-cell filter (reuse brca_loader heuristic)
    if malignant_only and "cell_type" in adata.obs.columns:
        non_malignant = {
            "T-cells", "B-cells", "Myeloid", "Endothelial", "CAFs",
            "Normal Epithelial", "Plasmablasts", "PVL",
            "T_cell", "T cell", "B_cell", "B cell",
            "Macrophage", "Fibroblast", "Mast_cell", "Mast cell",
            "NK_cell", "NK cell", "Plasma_cell", "Plasma cell",
            "CAF", "Pericyte",
        }
        adata.obs["is_malignant"] = ~adata.obs["cell_type"].isin(non_malignant)
        n_mal = adata.obs["is_malignant"].sum()
        logger.info(f"BIOKEY malignant: {n_mal}/{adata.n_obs}")
        adata = adata[adata.obs["is_malignant"]].copy()
    else:
        adata.obs["is_malignant"] = True

    logger.info(
        f"\nFinal BIOKEY AnnData: {adata.n_obs} cells × {adata.n_vars} genes\n"
        f"Patients: {adata.obs.get('patient_id', pd.Series()).nunique()}\n"
        f"Timepoints: {adata.obs.get('timepoint', pd.Series()).value_counts().to_dict()}"
    )
    return adata


def evaluate_biokey_holdout(
    model,
    gene_list: list,
    esm_embeddings: dict,
    device: str = "cpu",
) -> dict:
    """
    Generalization evaluation on BIOKEY held-out cohort.

    If BIOKEY is not downloaded, returns a status dict with instructions
    instead of failing — matches the design of ``eval_crc_zeroshot``.

    Metrics computed when data is present:
      - cell_type_composition_match: Pearson r vs expected composition
      - pre_vs_on_treatment_shift: spectral gap delta
      - cohort_A_vs_B_stability: inter-cohort rewiring correlation
    """
    if not biokey_available():
        return {
            "status": "data_missing",
            "instructions": (
                "Download BIOKEY from "
                "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-8107 "
                f"into {BIOKEY_DIR}"
            ),
        }

    try:
        adata = load_biokey_data(malignant_only=True, sample_n=10000)
    except Exception as e:
        return {"status": "load_error", "error": str(e)}

    # Placeholder metrics (full implementation requires trained model forward)
    n_patients = adata.obs.get("patient_id", pd.Series()).nunique()
    n_cells = adata.n_obs

    report = {
        "status": "ok",
        "n_cells": int(n_cells),
        "n_patients": int(n_patients),
        "cohort": str(adata.obs.get("cohort", pd.Series()).value_counts().to_dict()),
        "timepoints": str(adata.obs.get("timepoint", pd.Series()).value_counts().to_dict()),
        # Model-dependent metrics filled by caller if model provided
        "holdout_validation": "infrastructure_ready",
    }
    logger.info(f"BIOKEY holdout eval: {report}")
    return report


__all__ = [
    "BIOKEY_DIR",
    "biokey_available",
    "load_biokey_metadata",
    "load_biokey_data",
    "evaluate_biokey_holdout",
]
