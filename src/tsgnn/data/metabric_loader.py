"""
METABRIC bulk-expression + mutation + clinical loader.

Used for Nature claim P3: bulk tumors harboring p53-mutant alleles show
enrichment (BH-corrected) of genes predicted downstream by TS-GNN.

Builds AnnData where:
    .X              — sample × gene illumina-microarray expression
    .var_names      — Hugo gene symbol
    .obs            — patient + sample clinical annotations
    .obs["tp53_mutation"]    — HGVSp_Short (e.g. p.R175H) or "WT"
    .obs["tp53_allele_canonical"]  — canonical hotspot (R175H/R273H/…) or "WT"/"other"
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .download import RAW_DIR

logger = logging.getLogger(__name__)

METABRIC_DIR = RAW_DIR / "metabric"
METABRIC_EXPR = METABRIC_DIR / "data_mrna_illumina_microarray.txt"
METABRIC_PATIENT = METABRIC_DIR / "data_clinical_patient.txt"
METABRIC_SAMPLE = METABRIC_DIR / "data_clinical_sample.txt"
METABRIC_MUT = METABRIC_DIR / "data_mutations.txt"

HOTSPOTS = {"R175H", "R273H", "R248W", "R248Q", "R282W", "G245S", "Y220C"}


def metabric_available() -> bool:
    return all(p.exists() for p in (METABRIC_EXPR, METABRIC_PATIENT, METABRIC_SAMPLE, METABRIC_MUT))


def _load_expression() -> pd.DataFrame:
    """Return (genes × samples) expression DataFrame indexed by Hugo_Symbol."""
    df = pd.read_csv(METABRIC_EXPR, sep="\t", comment="#", low_memory=False)
    df = df.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in df.columns])
    df = df.dropna(subset=["Hugo_Symbol"])
    df = df.groupby("Hugo_Symbol", sort=False).mean(numeric_only=True)
    return df


def _load_clinical() -> pd.DataFrame:
    """Merge patient + sample clinical tables on Patient Identifier."""
    pt = pd.read_csv(METABRIC_PATIENT, sep="\t", comment="#", low_memory=False)
    sm = pd.read_csv(METABRIC_SAMPLE, sep="\t", comment="#", low_memory=False)
    key = "PATIENT_ID" if "PATIENT_ID" in sm.columns else "Patient Identifier"
    clin = sm.merge(pt, on=key, how="left", suffixes=("", "_pt"))
    sid_col = "SAMPLE_ID" if "SAMPLE_ID" in clin.columns else "Sample Identifier"
    clin = clin.set_index(sid_col)
    return clin


_HGVS_RE = re.compile(r"p\.([A-Z])(\d+)([A-Z])$")


def _canonicalize(hgvs: str) -> str:
    """p.R175H → R175H. Non-hotspot/insertion/frameshift → 'other'."""
    if not isinstance(hgvs, str):
        return "WT"
    m = _HGVS_RE.match(hgvs.strip())
    if not m:
        return "other"
    short = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return short if short in HOTSPOTS else "other"


def _load_tp53_mutations() -> pd.DataFrame:
    """One row per sample → most-damaging TP53 HGVS call (or absent)."""
    muts = pd.read_csv(METABRIC_MUT, sep="\t", comment="#", low_memory=False,
                       usecols=["Hugo_Symbol", "HGVSp_Short", "Variant_Classification",
                                "Tumor_Sample_Barcode"])
    tp = muts[muts["Hugo_Symbol"] == "TP53"].copy()
    # Priority: missense hotspot > other protein-altering > silent
    priority = {"Missense_Mutation": 0, "Nonsense_Mutation": 1, "In_Frame_Ins": 2,
                "In_Frame_Del": 2, "Frame_Shift_Ins": 3, "Frame_Shift_Del": 3,
                "Splice_Site": 4, "Silent": 9}
    tp["__pri"] = tp["Variant_Classification"].map(priority).fillna(5)
    tp = tp.sort_values(["Tumor_Sample_Barcode", "__pri"])
    tp = tp.drop_duplicates("Tumor_Sample_Barcode", keep="first")
    tp = tp.set_index("Tumor_Sample_Barcode")[["HGVSp_Short", "Variant_Classification"]]
    tp["tp53_allele_canonical"] = tp["HGVSp_Short"].map(_canonicalize)
    return tp


def load_metabric_data(
    require_mutation_call: bool = False,
    hotspots_only: bool = False,
) -> "anndata.AnnData":
    """Return AnnData (samples × genes) with TP53 annotation in obs.

    Args:
        require_mutation_call: drop samples not in mutation file (sequenced cohort).
        hotspots_only: keep only canonical hotspot mutants (R175H/R273H/...) + WT.
    """
    import anndata as ad

    expr = _load_expression()              # genes × samples
    clin = _load_clinical()                # samples × clinical fields
    tp53 = _load_tp53_mutations()          # samples × TP53 call

    sample_ids = [s for s in expr.columns if s in clin.index]
    logger.info("METABRIC: %d expression samples ∩ %d clinical → %d kept",
                expr.shape[1], len(clin), len(sample_ids))

    X = expr[sample_ids].T.to_numpy(dtype=np.float32)       # samples × genes
    obs = clin.loc[sample_ids].copy()
    obs["tp53_mutation"] = tp53["HGVSp_Short"].reindex(sample_ids).fillna("WT")
    obs["tp53_allele_canonical"] = tp53["tp53_allele_canonical"].reindex(sample_ids).fillna("WT")

    if require_mutation_call:
        keep = obs["tp53_mutation"].notna().values
        X, obs = X[keep], obs.iloc[keep]
    if hotspots_only:
        keep = obs["tp53_allele_canonical"].isin(HOTSPOTS | {"WT"}).values
        X, obs = X[keep], obs.iloc[keep]

    var = pd.DataFrame(index=pd.Index(expr.index, name="Hugo_Symbol"))
    adata = ad.AnnData(X=X, obs=obs.reset_index(), var=var)
    adata.obs_names = obs.index.astype(str)
    adata.obs_names_make_unique()
    return adata


def allele_cohorts(adata) -> dict:
    """Split-indices by canonical allele for downstream enrichment."""
    out: dict = {}
    for allele, sub in adata.obs.groupby("tp53_allele_canonical"):
        out[str(allele)] = sub.index.tolist()
    return out
