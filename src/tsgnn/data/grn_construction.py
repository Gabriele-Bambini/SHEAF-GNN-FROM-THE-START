"""
Base GRN construction from multi-omics priors.

Phase 1E.4-5: Build the directed regulatory graph from multiple evidence layers.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

from tsgnn.data import EXTERNAL_DIR


def load_scenic_regulons(regulon_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load pySCENIC regulon results.

    pySCENIC must be run externally (it takes 2-4 hours).
    Expected input: adjacencies CSV from GRNBoost2 + motif-pruned regulons.

    Returns:
        Dict mapping TF name to list of target gene names.
    """
    if regulon_path and regulon_path.exists():
        adj = pd.read_csv(regulon_path, sep="\t")
        regulons = {}
        for tf in adj["TF"].unique():
            targets = adj[adj["TF"] == tf]["target"].tolist()
            regulons[tf] = targets
        logger.info(f"Loaded {len(regulons)} regulons from pySCENIC")
        return regulons

    logger.warning(
        "pySCENIC regulons not found. To generate:\n"
        "  1. pip install pyscenic\n"
        "  2. pyscenic grn <expr_matrix> <tf_list> -o adj.tsv\n"
        "  3. pyscenic ctx adj.tsv <motif_db> --annotations_fname <motif_annot> -o reg.csv\n"
        "  4. Place adj.tsv in data/processed/scenic_adjacencies.tsv"
    )
    return {}


def load_regnetwork(path: Optional[Path] = None) -> pd.DataFrame:
    """Load RegNetwork TF-target relationships."""
    rn_dir = EXTERNAL_DIR / "regnetwork"
    if path is None:
        # human_regulatory.txt is the canonical name; human.source is the
        # alternative name produced by the RegNetwork 2.0 zip download.
        candidates = [
            rn_dir / "human_regulatory.txt",
            rn_dir / "human.source",
        ]
        path = next((p for p in candidates if p.exists()), candidates[0])

    if path.exists():
        df = pd.read_csv(path, sep="\t", header=None, names=["source", "target", "type"])
        logger.info(f"Loaded {len(df)} RegNetwork edges from {path.name}")
        return df

    logger.warning(f"RegNetwork file not found at {path}")
    return pd.DataFrame(columns=["source", "target", "type"])


def load_targetgenereg(path: Optional[Path] = None) -> pd.DataFrame:
    """Load TargetGeneReg 2.0 p53 target genes."""
    if path is None:
        tgr_dir = EXTERNAL_DIR / "targetgenereg"
        candidates = [
            tgr_dir / "p53_targets_benchmark.xlsx",
            tgr_dir / "p53_targets_benchmark.csv",
            tgr_dir / "p53_targets.csv",
        ]
        path = next((p for p in candidates if p.exists()), candidates[0])

    if path.exists():
        if path.suffix == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        # Normalize column names for downstream compatibility
        col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=col_map)
        if "gene_name" not in df.columns:
            # Try common alternatives
            for col in df.columns:
                if "gene" in col.lower() or "symbol" in col.lower():
                    df = df.rename(columns={col: "gene_name"})
                    break
        logger.info(f"Loaded {len(df)} p53 targets from TargetGeneReg")
        return df

    logger.warning(f"TargetGeneReg file not found at {path}")
    return pd.DataFrame(columns=["gene_name", "regulation_direction", "evidence_level"])


# ── BRCA-specific key transcription factors (Task 2.2) ─────────────────────
BRCA_KEY_TFS = {
    "FOXA1",   # master regulator of luminal BRCA identity
    "GATA3",   # luminal differentiation
    "ESR1",    # estrogen receptor — primary luminal driver
    "RUNX1",   # BRCA tumor suppressor
    "MYC",     # amplified in BRCA subtypes
    "E2F1",    # cell cycle regulator, TP53 target
    "NFKB1",   # inflammatory signalling in tumors
    "STAT3",   # JAK-STAT signalling
    "YAP1",    # Hippo pathway effector
    "TEAD4",   # YAP1 co-activator
}


def _build_string_id_map(alias_path: Path, target_species: int = 9606) -> dict:
    """Map Ensembl protein ID (9606.ENSP...) → gene symbol.

    Reads the STRING alias file (v12.0) and returns a dictionary
    {protein_id: gene_symbol} using only high-confidence gene-symbol sources.
    Uses a generator for memory efficiency.
    """
    import gzip
    id_map = {}
    try:
        logger.info(f"Building local STRING alias map from {alias_path.name}...")
        # Restrict to reliable gene-symbol sources
        symbol_sources = {"BioMart_HUGO", "Ensembl_HGNC", "BLAST_UniProt_GN", "Ensembl_gene_name"}
        species_prefix = f"{target_species}."
        
        with gzip.open(alias_path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                protein_id, alias, source = parts[0], parts[1], parts[2]
                
                if protein_id.startswith(species_prefix) and source in symbol_sources:
                    # Keep the first symbol encountered for each protein ID
                    if protein_id not in id_map:
                        id_map[protein_id] = alias
        
        logger.info(f"STRING alias map built: {len(id_map):,} protein → gene mappings")
        return id_map
    except Exception as e:
        logger.warning(f"Could not build STRING alias map: {e}")
        return {}


def load_string_ppi(
    path: Optional[Path] = None,
    min_score: int = 700,
    tf_list: Optional[set] = None,
) -> pd.DataFrame:
    """Load STRING PPI network, filtered to TFs and high-confidence interactions.

    Strictly uses local alias files (9606.protein.aliases.v12.0.txt.gz) for
    ID mapping to ensure HPC portability without internet access.
    """
    if path is None:
        path = EXTERNAL_DIR / "string" / "9606.protein.links.v12.0.txt.gz"

    if path.exists():
        logger.info("Loading STRING PPI (this may take a moment)...")
        compression = "gzip" if str(path).endswith(".gz") else "infer"
        df = pd.read_csv(path, sep=" ", compression=compression)
        df = df[df["combined_score"] >= min_score]
        logger.info(f"Loaded {len(df)} STRING interactions (score >= {min_score}) before ID mapping")

        # Apply alias mapping: look for any aliases file in the same directory
        alias_path = path.parent / "9606.protein.aliases.v12.0.txt.gz"
        if alias_path.exists():
            id_map = _build_string_id_map(alias_path)
            df["gene1"] = df["protein1"].map(id_map)
            df["gene2"] = df["protein2"].map(id_map)
            df = df.dropna(subset=["gene1", "gene2"])
            logger.info(
                f"After gene symbol mapping: {len(df)} STRING interactions with known gene IDs"
            )
            if tf_list:
                mask = df["gene1"].isin(tf_list) | df["gene2"].isin(tf_list)
                df = df[mask]
                logger.info(f"After TF filtering: {len(df)} TF-involving interactions")
        else:
            logger.warning(
                f"STRING alias file not found at {alias_path}. PPI mapping skipped."
            )
            df["gene1"] = pd.NA
            df["gene2"] = pd.NA

        return df

    logger.warning(f"STRING file not found at {path}")
    return pd.DataFrame()


def compute_spearman_weights(
    expr_matrix: np.ndarray,
    edge_index: torch.Tensor,
    gene_list: List[str],
) -> torch.Tensor:
    """
    Compute Spearman correlation weights for prior-constrained edges.

    Implements A = M ⊙ R where M is the binary prior mask (edge_index)
    and R is the empirical Spearman correlation.

    Fully vectorized: rank-transforms the entire expression matrix once,
    then computes Pearson on ranks (= Spearman) for all edges simultaneously.
    O(n_cells * N + E) instead of the previous O(E * n_cells) loop.

    Args:
        expr_matrix: (n_cells, n_genes) normalized expression matrix.
        edge_index: (2, E) prior-constrained edge indices.
        gene_list: List of gene names (for logging only).

    Returns:
        edge_weights: (E,) Spearman correlation for each prior edge.
    """
    from scipy.stats import rankdata

    E = edge_index.shape[1]
    n_cells, n_genes = expr_matrix.shape
    weights = torch.zeros(E, dtype=torch.float32)

    src_idx = edge_index[0].numpy()
    tgt_idx = edge_index[1].numpy()

    # Filter edges with valid gene indices
    valid = (src_idx < n_genes) & (tgt_idx < n_genes)
    v_src = src_idx[valid]
    v_tgt = tgt_idx[valid]

    if len(v_src) == 0:
        logger.warning("No valid edges for Spearman computation.")
        return weights

    # Rank-transform every gene column once — O(n_cells * n_genes)
    ranked = np.apply_along_axis(rankdata, 0, expr_matrix).astype(np.float32)

    # Extract source and target columns for all valid edges — O(E)
    x = ranked[:, v_src]   # (n_cells, E_valid)
    y = ranked[:, v_tgt]   # (n_cells, E_valid)

    # Pearson on ranks = Spearman — fully vectorized
    x_c = x - x.mean(axis=0)
    y_c = y - y.mean(axis=0)
    cov = (x_c * y_c).mean(axis=0)
    std_prod = x.std(axis=0) * y.std(axis=0)

    rho = np.where(std_prod > 1e-8, cov / std_prod, 0.0)
    rho = np.where(np.isfinite(rho), rho, 0.0)

    weights[torch.from_numpy(valid)] = torch.from_numpy(rho).float()

    n_neg = (weights < 0).sum().item()
    n_pos = (weights > 0).sum().item()
    logger.info(
        f"Spearman weights computed (vectorized): {n_pos} positive, {n_neg} negative "
        f"over {valid.sum()}/{E} valid edges (neg = repression)"
    )
    return weights


def load_dorothea_prior_edges(gene_list: List[str], levels: List[str] = None) -> dict:
    """
    Load DoRothEA TF-target edges as highest-confidence GRN priors.

    DoRothEA (Garcia-Alonso et al. 2019, Genome Research) integrates:
    - ChIP-seq binding peaks (experimental)
    - TF binding motifs in promoters (computational)
    - Literature-curated TF-target interactions
    - Gene co-expression

    Confidence levels A+B have the highest experimental support.

    Args:
        gene_list: Gene symbols in the model.
        levels: DoRothEA confidence levels ['A', 'B', 'C', 'D'] (default ['A', 'B']).

    Returns:
        edges: Dict {(src_idx, tgt_idx): weight} with signed weights
               (positive=activation, negative=repression).
        Empty dict if decoupler unavailable.
    """
    if levels is None:
        levels = ['A', 'B']

    gene_set = set(gene_list)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    edges = {}

    try:
        import decoupler as dc
        # Support multiple decoupler API versions:
        # < 1.3:  dc.get_dorothea(organism, levels)
        # 1.3-1.5: dc.get_resource('DoRothEA', organism)
        # >= 1.6:  dc.op.dorothea.dorothea() or decoupler.datasets
        dorothea = None
        if hasattr(dc, 'get_dorothea'):
            dorothea = dc.get_dorothea(organism='human', levels=levels)
        elif hasattr(dc, 'get_resource'):
            dorothea = dc.get_resource('DoRothEA', organism='human')
            if 'confidence' in dorothea.columns:
                dorothea = dorothea[dorothea['confidence'].isin(levels)]
            if 'source' not in dorothea.columns and 'tf' in dorothea.columns:
                dorothea = dorothea.rename(columns={'tf': 'source', 'gene': 'target'})
        else:
            # decoupler >= 1.6: try datasets or op submodules
            try:
                import decoupler.datasets as dcd
                dorothea = dcd.dorothea(organism='human')
                if 'confidence' in dorothea.columns:
                    dorothea = dorothea[dorothea['confidence'].isin(levels)]
            except Exception:
                raise ImportError("Cannot find DoRothEA loader in this decoupler version")
        if dorothea is None:
            raise ImportError("DoRothEA not loadable")
        logger.info(f"DoRothEA loaded: {len(dorothea)} TF-target interactions (levels {levels})")

        n_added = 0
        for _, row in dorothea.iterrows():
            src_gene = row['source']
            tgt_gene = row['target']
            weight = float(row.get('weight', 1.0))

            if src_gene in gene_set and tgt_gene in gene_set and src_gene != tgt_gene:
                src_idx = gene_to_idx[src_gene]
                tgt_idx = gene_to_idx[tgt_gene]
                # Keep the interaction with highest absolute weight if seen before
                existing = edges.get((src_idx, tgt_idx), 0.0)
                if abs(weight) > abs(existing):
                    edges[(src_idx, tgt_idx)] = weight
                n_added += 1

        logger.info(f"DoRothEA prior edges mapped to model genes: {len(edges)}")
        return edges

    except ImportError:
        logger.warning("decoupler not installed; skipping DoRothEA priors. pip install decoupler")
        return {}
    except Exception as e:
        logger.warning(f"DoRothEA loading failed: {e}; skipping DoRothEA priors.")
        return {}


def construct_base_grn(
    gene_list: List[str],
    scenic_regulons: Optional[Dict[str, List[str]]] = None,
    regnetwork_edges: Optional[pd.DataFrame] = None,
    targetgenereg: Optional[pd.DataFrame] = None,
    string_ppi: Optional[pd.DataFrame] = None,
    expr_matrix: Optional[np.ndarray] = None,
    scprint_edges: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Construct the base GRN from multiple evidence layers.

    Topology (M): determined by prior databases (SCENIC, RegNetwork, TargetGeneReg).
    Weights (R): if expr_matrix is provided, edges are weighted by Spearman
                 correlation (A = M ⊙ R), preserving negative values for
                 repressive interactions. Otherwise, uses fixed prior weights.

    Layers (priority order):
    1. pySCENIC regulons (data-driven TF -> target with importance scores)
    2. RegNetwork + TargetGeneReg 2.0 prior edges
    3. STRING PPI for TF-TF co-regulatory edges

    Args:
        gene_list: List of N selected gene names (defines node ordering).
        scenic_regulons: Dict[TF -> List[target]] from pySCENIC.
        regnetwork_edges: DataFrame of regulatory edges.
        targetgenereg: DataFrame of p53 targets.
        string_ppi: DataFrame of PPI edges.
        expr_matrix: Optional (n_cells, N) expression matrix for Spearman weighting.

    Returns:
        edge_index: (2, E) tensor of directed edges
        edge_weight: (E,) tensor of edge weights (Spearman if expr_matrix provided)
        edge_metadata: Dict with edge annotations
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    N = len(gene_list)
    edges = {}  # (src_idx, tgt_idx) -> weight
    edge_sources = {}  # (src_idx, tgt_idx) -> source layer name

    # ── Layer 0.5: scPRINT Foundation Model ──
    if scprint_edges is not None:
        src_nodes, tgt_nodes = scprint_edges
        n_scprint = 0
        for u, v in zip(src_nodes.tolist(), tgt_nodes.tolist()):
            if (u, v) not in edges:
                edges[(u, v)] = 1.0  # High confidence
                edge_sources[(u, v)] = "scPRINT"
                n_scprint += 1
        logger.info(f"Layer 0.5 (scPRINT): {n_scprint} high-confidence data-driven edges added")

    # ── Layer 1: pySCENIC regulons ──
    if scenic_regulons:
        n_scenic = 0
        for tf, targets in scenic_regulons.items():
            if tf not in gene_to_idx:
                continue
            src = gene_to_idx[tf]
            for target in targets:
                if target not in gene_to_idx:
                    continue
                tgt = gene_to_idx[target]
                if src != tgt:
                    edges[(src, tgt)] = max(edges.get((src, tgt), 0), 1.0)
                    edge_sources[(src, tgt)] = "SCENIC"
                    n_scenic += 1
        logger.info(f"Layer 1 (SCENIC): {n_scenic} edges added")

    # ── Layer 2: RegNetwork prior edges ──
    if regnetwork_edges is not None and len(regnetwork_edges) > 0:
        n_reg = 0
        for _, row in regnetwork_edges.iterrows():
            src_name = str(row.get("source", ""))
            tgt_name = str(row.get("target", ""))
            if src_name in gene_to_idx and tgt_name in gene_to_idx:
                src = gene_to_idx[src_name]
                tgt = gene_to_idx[tgt_name]
                if src != tgt and (src, tgt) not in edges:
                    edges[(src, tgt)] = 0.5  # Lower confidence than SCENIC
                    edge_sources[(src, tgt)] = "RegNetwork"
                    n_reg += 1
        logger.info(f"Layer 2 (RegNetwork): {n_reg} new edges added")

    # ── Layer 3: TargetGeneReg 2.0 p53 targets ──
    if targetgenereg is not None and len(targetgenereg) > 0:
        n_tgr = 0
        tp53_idx = gene_to_idx.get("TP53")
        if tp53_idx is not None:
            for _, row in targetgenereg.iterrows():
                target_name = str(row.get("gene_name", ""))
                if target_name in gene_to_idx:
                    tgt = gene_to_idx[target_name]
                    direction = row.get("regulation_direction", "activated")
                    weight = 1.5 if direction == "activated" else -1.5
                    edges[(tp53_idx, tgt)] = weight
                    edge_sources[(tp53_idx, tgt)] = "TargetGeneReg"
                    n_tgr += 1
            logger.info(f"Layer 3 (TargetGeneReg): {n_tgr} TP53 target edges added")
        else:
            logger.warning("TP53 not in gene list; cannot add TargetGeneReg edges")

    # ── Layer 4: STRING PPI (TF-TF co-regulatory, undirected) ──
    if string_ppi is not None and len(string_ppi) > 0:
        n_string = 0
        # After load_string_ppi() with alias mapping, gene1/gene2 columns are gene symbols
        if "gene1" in string_ppi.columns and "gene2" in string_ppi.columns and not string_ppi["gene1"].isna().all():
            for _, row in string_ppi.iterrows():
                g1 = str(row.get("gene1", ""))
                g2 = str(row.get("gene2", ""))
                if g1 in gene_to_idx and g2 in gene_to_idx:
                    i1 = gene_to_idx[g1]
                    i2 = gene_to_idx[g2]
                    if i1 != i2:
                        # Normalize score to [0, 1]
                        score = float(row.get("combined_score", 700)) / 1000.0
                        # Add both directions (co-regulatory, undirected)
                        if (i1, i2) not in edges:
                            edges[(i1, i2)] = score
                            edge_sources[(i1, i2)] = "STRING"
                            n_string += 1
                        if (i2, i1) not in edges:
                            edges[(i2, i1)] = score
                            edge_sources[(i2, i1)] = "STRING"
                            n_string += 1
        else:
            logger.info("STRING alias mapping skipped (alias file missing).")
                    
        if n_string > 0:
            logger.info(f"Layer 4 (STRING): {n_string} TF-TF edges added")
        else:
            logger.info("Layer 4 (STRING): no edges added (alias mapping may be missing)")

    # Require at least some prior edges — synthetic fallback is never used on real data
    if len(edges) == 0:
        raise RuntimeError(
            "No prior regulatory edges found in any database. "
            "Ensure at least one of the following is available in TSGNN_DATA_DIR:\n"
            "  - SCENIC adjacency matrix (adj.tsv / pyscenic_grn_adj.tsv)\n"
            "  - RegNetwork database files (network.csv)\n"
            "  - TargetGeneReg database (TargetGeneReg.csv)\n"
            "  - STRING protein interaction network (9606.protein.links.v12.0.txt.gz)\n"
            "Run `python scripts/download_all_datasets.py` to fetch all databases."
        )

    # Convert to tensors
    edge_list = sorted(edges.keys())
    src_indices = [e[0] for e in edge_list]
    tgt_indices = [e[1] for e in edge_list]
    prior_weights = [edges[e] for e in edge_list]

    edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)

    # ── Spearman Correlation Weighting (A = M ⊙ R) ──
    if expr_matrix is not None:
        logger.info("Computing Spearman correlation weights (A = M ⊙ R)...")
        edge_weight = compute_spearman_weights(expr_matrix, edge_index, gene_list)
    else:
        edge_weight = torch.tensor(prior_weights, dtype=torch.float32)
        logger.info("Using fixed prior weights (no expression matrix provided for Spearman)")

    logger.info(f"Base GRN constructed: {N} nodes, {edge_index.shape[1]} edges")

    edge_metadata = {
        "gene_list": gene_list,
        "gene_to_idx": gene_to_idx,
        "n_nodes": N,
        "n_edges": edge_index.shape[1],
        "edge_sources": edge_sources,
        "uses_spearman": expr_matrix is not None,
    }

    return edge_index, edge_weight, edge_metadata


def _generate_synthetic_grn(n_nodes: int, target_edges: int = 5000, seed: int = 42) -> Dict:
    """Generate a synthetic scale-free GRN for development/testing.

    Every stochastic step uses the same seeded `numpy.random.Generator`, so two
    calls with the same `seed` return identical edge sets and weights — a
    prerequisite for deterministic unit tests and ablation runs.
    """
    import networkx as nx

    rng = np.random.default_rng(seed)

    # Scale-free graph (regulatory networks are approximately scale-free)
    m = max(1, min(target_edges // n_nodes, n_nodes - 1))
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    G = G.to_directed()

    edges = {}
    for u, v in G.edges():
        weight = float(rng.uniform(0.1, 2.0))
        # ~20% of edges are repressive (negative weight)
        if rng.random() < 0.2:
            weight = -weight
        edges[(u, v)] = weight

    # Trim to target using the same seeded generator
    if len(edges) > target_edges:
        keys = list(edges.keys())
        rng.shuffle(keys)
        edges = {k: edges[k] for k in keys[:target_edges]}

    logger.info(f"Synthetic GRN: {n_nodes} nodes, {len(edges)} edges")
    return edges


# ── Public API alias ────────────────────────────────────────────────────────

build_base_grn = construct_base_grn  # backwards-compatible alias


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Test with synthetic data
    gene_list = [f"GENE_{i}" for i in range(500)]
    gene_list[0] = "TP53"
    edge_index, edge_weight, meta = construct_base_grn(gene_list)
    print(f"GRN: {meta['n_nodes']} nodes, {meta['n_edges']} edges")
    print(f"Edge weight range: [{edge_weight.min():.2f}, {edge_weight.max():.2f}]")
    print(f"Repressive edges: {(edge_weight < 0).sum()}/{len(edge_weight)}")
