"""
Ovarian cancer (HGSOC) scRNA-seq loader for cross-cancer zero-shot validation.

Reference
---------
Izar B. et al. (2020) "A single-cell landscape of high-grade serous ovarian cancer."
Nature Medicine 26, 1271-1279.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146026

Dataset characteristics
-----------------------
~11k single cells from 8 HGSOC patients (ascites + primary tumour).
Profiled by 10x Chromium and SmartSeq2.
HGSOC has ~96% TP53 mutation prevalence → ideal cross-tissue TP53-signal test.

Expected files (genes × cells TSV, log-normalised)
-------------------
data/raw/ovarian/GSE146026_Izar_HGSOC_ascites_10x_log.tsv.gz  (98 MB, primary)
data/raw/ovarian/GSE146026_Izar_HGSOC_ascites_SS2_log.tsv.gz  (54 MB, secondary)
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tsgnn.data import RAW_DIR

logger = logging.getLogger(__name__)

OVARIAN_DIR = RAW_DIR / "ovarian"

# 10x preferred; SS2 as fallback
_CANDIDATES = [
    OVARIAN_DIR / "GSE146026_Izar_HGSOC_ascites_10x_log.tsv.gz",
    OVARIAN_DIR / "GSE146026_Izar_HGSOC_ascites_10x_log.tsv",
    OVARIAN_DIR / "GSE146026_Izar_HGSOC_ascites_SS2_log.tsv.gz",
    OVARIAN_DIR / "GSE146026_Izar_HGSOC_ascites_SS2_log.tsv",
]


def _find_tsv() -> Optional[Path]:
    OVARIAN_DIR.mkdir(parents=True, exist_ok=True)
    for p in _CANDIDATES:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    # Any tsv fallback
    found = sorted(OVARIAN_DIR.glob("*.tsv*"))
    return found[0] if found else None


def ovarian_available() -> bool:
    return _find_tsv() is not None


def _read_tsv_matrix(path: Path) -> tuple[np.ndarray, list, list]:
    """Read GEO TSV: cells × genes (log-normalised, gzipped).

    Izar 2020 layout: rows=cells (with barcode in col 0), cols=genes.
    Returns (X: float32 cells×genes, gene_names, cell_barcodes).
    """
    logger.info(f"Reading {path.name} ({path.stat().st_size / 1e6:.0f} MB)...")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt") as fh:
        # Header = gene names (skip col 0)
        header = fh.readline().rstrip("\n").split("\t")
        genes = header[1:]  # Skip cell barcode column label

        # Data rows: [cell_barcode, expr1, expr2, ...]
        cell_ids = []
        data_rows = []
        n_skip = 0
        for line_num, line in enumerate(fh, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                n_skip += 1
                continue
            try:
                vals = [float(v) for v in parts[1:]]
                cell_ids.append(parts[0])
                data_rows.append(vals)
            except ValueError:
                n_skip += 1
                continue

    X = np.array(data_rows, dtype=np.float32)  # shape: (n_cells, n_genes)
    logger.info(f"Matrix: {X.shape[0]} cells × {X.shape[1]} genes (skipped {n_skip})")
    return X, genes, cell_ids


def load_ovarian_data(
    sample_n: Optional[int] = 10000,
    random_state: int = 42,
    malignant_only: bool = True,
) -> "anndata.AnnData":
    """Load HGSOC scRNA-seq (Izar 2020, GSE146026) into AnnData.

    Parameters
    ----------
    sample_n : int, optional
        Subsample cells (default 10k). None = all.
    malignant_only : bool
        Izar 2020 has no cell-type labels in the TSV; all cells are from
        EPCAM+ sorted or CD45- depleted ovarian tumour material → effectively
        all malignant. This flag is a no-op here but kept for API compatibility.
    """
    try:
        import anndata
    except ImportError as e:
        raise ImportError("anndata required: pip install anndata") from e

    tsv_path = _find_tsv()
    if tsv_path is None:
        raise FileNotFoundError(
            f"Ovarian TSV not found in {OVARIAN_DIR}.\n"
            "Run on HPC: bash hpc/download_ovarian_data.sh"
        )

    X, genes, cells = _read_tsv_matrix(tsv_path)

    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    logger.info(f"AnnData: {adata.n_obs} cells x {adata.n_vars} genes")

    # All cells from this dataset are tumour-derived (EPCAM+/CD45-)
    adata.obs["is_malignant"] = True

    # No per-cell TP53 allele → default "unknown" (same as CRC)
    adata.obs["tp53_allele"] = "unknown"

    if sample_n is not None and sample_n < adata.n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=sample_n, replace=False)
        adata = adata[idx].copy()
        logger.info(f"Subsampled to {adata.n_obs} cells")

    return adata


__all__ = ["OVARIAN_DIR", "ovarian_available", "load_ovarian_data"]
