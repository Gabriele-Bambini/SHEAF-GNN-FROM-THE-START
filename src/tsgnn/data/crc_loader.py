"""
CRC (GSE178341, Pelka et al. 2021) scRNA-seq loader for cross-cancer
zero-shot validation.

Reference
---------
Pelka K. et al. (2021) "Spatially organized multicellular immune hubs in
human colorectal cancer." Cell 184, 4734-4752.e20.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178341

Dataset characteristics
-----------------------
~371k cells from 62 CRC patients (MMRd + MMRp). Used in TS-GNN as the
CROSS-CANCER zero-shot cohort: the model is trained on BRCA and evaluated
on CRC to measure tissue-agnostic TF-rewiring transferability.

Expected file
-------------
data/raw/crc/GSE178341_crc10x_full_c295v4_submit.h5  (HDF5, ~1.2 GB)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tsgnn.data import RAW_DIR

logger = logging.getLogger(__name__)

CRC_DIR = RAW_DIR / "crc"
CRC_H5  = CRC_DIR / "GSE178341_crc10x_full_c295v4_submit.h5"


def crc_available() -> bool:
    return CRC_H5.exists() and CRC_H5.stat().st_size > 1_000_000


def load_crc_data(
    sample_n: Optional[int] = 20000,
    random_state: int = 42,
    malignant_only: bool = True,
) -> "anndata.AnnData":
    """Load CRC scRNA-seq from GSE178341 HDF5 into AnnData.

    The Pelka 2021 H5 stores a standard 10x-style matrix (genes × cells)
    plus per-cell metadata tables.  We use h5py for raw access and then
    wrap into AnnData.

    Parameters
    ----------
    sample_n : int, optional
        Subsample cells (default 20k — the full 371k cells rarely fit in RAM
        during zero-shot eval). Pass ``None`` for full load.
    malignant_only : bool
        If True, retain only cells labeled as tumor/epithelial malignant.
    """
    try:
        import anndata
        import h5py
        import scipy.sparse as sp
    except ImportError as e:
        raise ImportError(
            "CRC loader requires h5py + anndata + scipy. "
            "pip install h5py anndata scipy"
        ) from e

    if not crc_available():
        raise FileNotFoundError(
            f"CRC H5 not found at {CRC_H5}. "
            "Run tsgnn.data.download.download_crc_scrna() first."
        )

    logger.info(f"Opening {CRC_H5} ({CRC_H5.stat().st_size / 1e9:.2f} GB)...")
    with h5py.File(str(CRC_H5), "r") as h5:
        # Discover matrix layout — Pelka uses "matrix" group with data/indices/indptr
        if "matrix" in h5:
            m = h5["matrix"]
            data    = m["data"][:]
            indices = m["indices"][:]
            indptr  = m["indptr"][:]
            shape   = tuple(m["shape"][:])  # (genes, cells)
            genes   = m["features"]["name"][:].astype(str)
            cells   = m["barcodes"][:].astype(str)
        else:
            # Fallback: simple "X" / "obs" / "var" layout
            data    = h5["X"]["data"][:]
            indices = h5["X"]["indices"][:]
            indptr  = h5["X"]["indptr"][:]
            shape   = tuple(h5["X"]["shape"][:])
            genes   = h5["var"]["gene"][:].astype(str) if "gene" in h5["var"] else np.array(
                [f"gene_{i}" for i in range(shape[0])])
            cells   = h5["obs"]["cell"][:].astype(str) if "cell" in h5["obs"] else np.array(
                [f"cell_{i}" for i in range(shape[1])])

        X = sp.csc_matrix((data, indices, indptr), shape=shape).tocsr().T  # → cells × genes

        # Optional: per-cell metadata (cluster, sample_id, etc.)
        obs = pd.DataFrame(index=cells)
        for key in ("clTopLevel", "clMidwayPr", "SPECIMEN_TYPE", "PatientTypeID",
                    "MMRStatus", "cell_type", "SampleID"):
            try:
                if key in h5:
                    vals = h5[key][:]
                    obs[key.lower()] = vals.astype(str) if vals.dtype.kind in "SOU" else vals
            except Exception:
                continue

    adata = anndata.AnnData(X=X.astype(np.float32),
                            obs=obs,
                            var=pd.DataFrame(index=genes))
    adata.var_names_make_unique()
    logger.info(f"Loaded CRC: {adata.n_obs} cells × {adata.n_vars} genes")

    # Subsample for zero-shot eval
    if sample_n is not None and sample_n < adata.n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=sample_n, replace=False)
        adata = adata[idx].copy()
        logger.info(f"Subsampled to {sample_n} cells")

    # Malignant filter (CRC-specific vocabulary from Pelka 2021)
    if malignant_only:
        mal_keys = ("cltoplevel", "clmidwaypr", "cell_type")
        mal_col = next((k for k in mal_keys if k in adata.obs.columns), None)
        if mal_col:
            mal_labels = {"Epi", "Tumor", "Malignant", "TC", "Epithelial"}
            adata.obs["is_malignant"] = adata.obs[mal_col].isin(mal_labels)
            n_mal = adata.obs["is_malignant"].sum()
            if n_mal > 0:
                logger.info(f"CRC malignant: {n_mal}/{adata.n_obs}")
                adata = adata[adata.obs["is_malignant"]].copy()
            else:
                logger.warning(f"No malignant cells found in {mal_col} — keeping all")
                adata.obs["is_malignant"] = True
        else:
            adata.obs["is_malignant"] = True

    # CRC has no TP53 allele per cell; default to "unknown" for zero-shot
    adata.obs["tp53_allele"] = "unknown"
    return adata


__all__ = ["CRC_DIR", "crc_available", "load_crc_data"]
