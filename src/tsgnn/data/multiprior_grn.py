"""
Multi-prior GRN construction.

Fuses three complementary sources into a single weighted TF→target edge list:
  - DoRothEA:  curated TF→target with mode-of-regulation and confidence (A..D)
  - JASPAR 2024: TF motif names (used as a mask: a TF is 'motif-supported' if
                 it appears in the JASPAR PFM set). Optional weight multiplier.
  - STRING v12: protein-protein combined_score ≥ threshold, mapped to Hugo
                symbols via the aliases table.

Output: pandas DataFrame with columns
    source | target | weight | evidence (comma-set of {"dorothea","string","jaspar"})

Design notes:
  - Hugo-symbol-first (matches METABRIC/AnnData var_names).
  - Deterministic: sorted edges, stable hashing.
  - Cached to disk as parquet at `data/processed/multiprior_grn.parquet`
    unless `force=True`.
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import DATA_DIR, EXTERNAL_DIR
PROCESSED_DIR = DATA_DIR / "processed"

logger = logging.getLogger(__name__)

DOROTHEA_CSV = EXTERNAL_DIR / "dorothea" / "dorothea_interactions.csv"
STRING_LINKS = EXTERNAL_DIR / "string" / "9606.protein.links.v12.0.txt"
STRING_ALIASES = EXTERNAL_DIR / "string" / "9606.protein.aliases.v12.0.txt.gz"
JASPAR_PFM = EXTERNAL_DIR / "jaspar" / "JASPAR2024_CORE_vertebrates.txt"

CACHE = PROCESSED_DIR / "multiprior_grn.parquet"

_DOROTHEA_CONF_W = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}


# ─────────────────────────── per-source loaders ──────────────────────────────

def _load_dorothea() -> pd.DataFrame:
    df = pd.read_csv(DOROTHEA_CSV)
    df = df.rename(columns={"TF": "source", "target": "target"})
    df["weight"] = df["confidence"].map(_DOROTHEA_CONF_W).fillna(0.1).astype(float)
    df["weight"] = df["weight"] * df["mor"].astype(float).abs().clip(lower=0.1)
    return df[["source", "target", "weight"]].assign(evidence="dorothea")


def _load_jaspar_tfs() -> set:
    """Return the set of Hugo TF symbols present in JASPAR CORE vertebrate PFMs."""
    tfs: set = set()
    with open(JASPAR_PFM, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith(">"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    # JASPAR TF names may be composite (e.g. "Ar::Foxa1"); split and upper.
                    for token in parts[1].replace("(", "::").replace(")", "").split("::"):
                        token = token.strip()
                        if token:
                            tfs.add(token.upper())
    return tfs


def _load_string_hugo(min_score: int = 700) -> pd.DataFrame:
    """STRING edges filtered by combined_score, mapped ENSP→Hugo symbol."""
    # Map ENSP → Hugo via aliases table (Ensembl_HGNC source preferred; fall back to BioMart_HUGO).
    logger.info("Loading STRING aliases (Hugo mapping) …")
    alias_rows = []
    preferred = {"Ensembl_HGNC", "BioMart_HUGO", "HGNC"}
    with gzip.open(STRING_ALIASES, "rt", encoding="utf-8", errors="ignore") as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if parts[2] in preferred:
                alias_rows.append((parts[0], parts[1].upper()))
    alias = pd.DataFrame(alias_rows, columns=["ensp", "hugo"]).drop_duplicates("ensp")
    e2h = dict(zip(alias["ensp"], alias["hugo"]))
    logger.info("  %d ENSP→Hugo mappings", len(e2h))

    logger.info("Loading STRING links (score ≥ %d) via chunked scan …", min_score)
    frames = []
    for chunk in pd.read_csv(STRING_LINKS, sep=" ",
                             usecols=["protein1", "protein2", "combined_score"],
                             chunksize=1_000_000):
        chunk = chunk[chunk["combined_score"] >= min_score]
        if chunk.empty:
            continue
        chunk["source"] = chunk["protein1"].map(e2h)
        chunk["target"] = chunk["protein2"].map(e2h)
        chunk = chunk.dropna(subset=["source", "target"])
        chunk = chunk[chunk["source"] != chunk["target"]]
        if not chunk.empty:
            chunk["weight"] = chunk["combined_score"].astype(float) / 1000.0
            frames.append(chunk[["source", "target", "weight"]])
    if not frames:
        return pd.DataFrame(columns=["source", "target", "weight", "evidence"])
    use = pd.concat(frames, ignore_index=True)
    logger.info("  STRING kept %d edges", len(use))
    return use.assign(evidence="string")


# ─────────────────────────── public API ──────────────────────────────────────

def build_multiprior_grn(
    genes: Optional[Iterable[str]] = None,
    priors: Iterable[str] = ("dorothea", "string", "jaspar"),
    string_min_score: int = 700,
    jaspar_boost: float = 1.25,
    force: bool = False,
) -> pd.DataFrame:
    """Construct a consensus weighted TF→target edge table.

    Args:
        genes: restrict edges to pairs where both endpoints are in this gene set.
        priors: any subset of {"dorothea","string","jaspar"}.
        string_min_score: STRING combined_score threshold (0–1000).
        jaspar_boost: multiplier applied to edges whose source is a JASPAR TF.
        force: bypass cache.

    Returns:
        DataFrame with cols [source, target, weight, evidence].
    """
    if CACHE.exists() and not force and genes is None and set(priors) == {"dorothea", "string", "jaspar"}:
        logger.info("Using cached multiprior GRN: %s", CACHE)
        return pd.read_parquet(CACHE)

    frames = []
    if "dorothea" in priors and DOROTHEA_CSV.exists():
        frames.append(_load_dorothea())
    if "string" in priors and STRING_LINKS.exists() and STRING_ALIASES.exists():
        frames.append(_load_string_hugo(min_score=string_min_score))

    if not frames:
        raise RuntimeError("No priors available — check data/external/ contents.")

    edges = pd.concat(frames, ignore_index=True)

    jaspar_tfs: set = set()
    if "jaspar" in priors and JASPAR_PFM.exists():
        jaspar_tfs = _load_jaspar_tfs()
        mask = edges["source"].str.upper().isin(jaspar_tfs)
        edges.loc[mask, "weight"] *= jaspar_boost
        edges.loc[mask, "evidence"] = edges.loc[mask, "evidence"] + ",jaspar"

    # Collapse duplicates by max-weight + union of evidence
    g = edges.groupby(["source", "target"], sort=False, as_index=False).agg(
        weight=("weight", "max"),
        evidence=("evidence", lambda s: ",".join(sorted(set(",".join(s).split(","))))),
    )

    if genes is not None:
        keep = set(g["source"]) & set(genes) if False else set(genes)  # explicit
        keep = set(genes)
        g = g[g["source"].isin(keep) & g["target"].isin(keep)].reset_index(drop=True)

    g = g.sort_values(["source", "target"]).reset_index(drop=True)

    if genes is None and set(priors) == {"dorothea", "string", "jaspar"}:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(CACHE, index=False)
        logger.info("Cached multiprior GRN → %s (%d edges)", CACHE, len(g))

    return g
