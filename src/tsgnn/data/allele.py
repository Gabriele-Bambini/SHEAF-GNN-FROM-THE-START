"""
TP53 allele extraction and ESM-2 embedding generation.

Phase 1D: Generate allele-specific protein embeddings from ESM-2.
Also handles allele assignment from WES/metadata to scRNA-seq cells.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

from tsgnn.data import DATA_DIR
ESM_DIR = DATA_DIR / "external" / "esm2_embeddings"

# ── TP53 wild-type protein sequence (UniProt P04637, canonical, 393 residues) ──
TP53_SEQUENCE = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
    "DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLNGTVNLPGRNSFEV"
    "RVCACPGRDRRTEEENLHKTTGIDSFLHSGAKVTCTYSPALNKMFCQLAKTCPVQLWVDST"
    "PPPGTRVRAMAIYKQSQHMTEVVRRCPHERCTPFHCDGAFHCPQSQSTRYHQTSCRHREK"
    "EILEGQGSCLQLTWKTSRSQNEFIPKRLCPELHQNTIPVTLSPDNLEVDVQHKTKRLKEI"
    "NLQHSSQGDQEQKDQSSTSRHKKLMFKTEGPDSD"
)

# Hotspot mutations: position (1-indexed), WT residue, mutant residue
TP53_HOTSPOT_MUTATIONS = {
    "R175H": (175, "R", "H"),
    "R273H": (273, "R", "H"),
    "R248W": (248, "R", "W"),
    "R248Q": (248, "R", "Q"),
    "R282W": (282, "R", "W"),
    "G245S": (245, "G", "S"),
    "Y220C": (220, "Y", "C"),
}

# Structural classification
CONTACT_MUTANTS = {"R273H", "R248W", "R248Q"}
CONFORMATIONAL_MUTANTS = {"R175H", "G245S", "R282W", "Y220C"}


def create_mutant_sequence(mutation: str) -> str:
    """Create mutant TP53 protein sequence by substituting the specified residue."""
    pos, wt_res, mut_res = TP53_HOTSPOT_MUTATIONS[mutation]
    seq = list(TP53_SEQUENCE)
    idx = pos - 1  # Convert to 0-indexed

    if idx >= len(seq):
        raise ValueError(f"Position {pos} is beyond sequence length {len(seq)}")
    if seq[idx] != wt_res:
        logger.warning(
            f"Expected {wt_res} at position {pos}, found {seq[idx]}. "
            "Sequence may differ from canonical. Proceeding with substitution."
        )

    seq[idx] = mut_res
    return "".join(seq)


def generate_esm2_embeddings(
    alleles: Optional[list] = None,
    model_name: str = "esm2_t33_650M_UR50D",
    device: Optional[str] = None,
    allow_placeholder: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generate ESM-2 embeddings for WT and all specified TP53 alleles.

    Args:
        alleles: List of allele names (e.g., ["R175H", "R273H"]).
                 If None, generates for all hotspot mutations.
        model_name: ESM-2 model to use.
        device: Torch device. Auto-detects CUDA if available.

    Returns:
        Dict mapping allele name to embedding tensor of shape (1280,).
        Also saves to data/external/esm2_embeddings/.
    """
    if alleles is None:
        alleles = list(TP53_HOTSPOT_MUTATIONS.keys())

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ESM_DIR.mkdir(parents=True, exist_ok=True)

    # Check if embeddings already exist
    all_exist = all((ESM_DIR / f"{a}.pt").exists() for a in alleles)
    if all_exist and (ESM_DIR / "WT.pt").exists():
        logger.info("All ESM-2 embeddings already cached. Loading...")
        embeddings = {"WT": torch.load(ESM_DIR / "WT.pt", weights_only=True)}
        for a in alleles:
            embeddings[a] = torch.load(ESM_DIR / f"{a}.pt", weights_only=True)
        return embeddings

    logger.info(f"Loading ESM-2 model: {model_name} on {device}")
    try:
        # Task 10.1: Cache ESM-2 on Drive between Colab sessions
        cache_dir = Path(os.environ.get("TSGNN_DATA_DIR", ".")).parent / "cache" / "esm2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir)
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()
    except ImportError:
        logger.error("fair-esm not installed. Install with: pip install fair-esm")
        if not allow_placeholder:
            raise RuntimeError(
                "fair-esm is not installed and strict scientific mode forbids placeholder "
                "ESM-2 embeddings. Precompute real embeddings in "
                f"{ESM_DIR} before running."
            )
        logger.info("Generating random placeholder embeddings for development...")
        return _generate_placeholder_embeddings(alleles)

    # Prepare sequences
    sequences = [("WT", TP53_SEQUENCE)]
    for allele in alleles:
        mut_seq = create_mutant_sequence(allele)
        sequences.append((allele, mut_seq))

    logger.info(f"Computing ESM-2 embeddings for {len(sequences)} sequences...")

    embeddings = {}
    # Process one at a time to manage memory
    for name, seq in sequences:
        batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])

        # Mean pool over residues (excluding BOS/EOS tokens)
        token_repr = results["representations"][33]
        # tokens: [BOS, residue_1, ..., residue_N, EOS]
        embedding = token_repr[0, 1:-1, :].mean(dim=0).cpu()  # (1280,)

        embeddings[name] = embedding
        torch.save(embedding, ESM_DIR / f"{name}.pt")
        logger.info(f"  {name}: embedding shape {embedding.shape}, norm {embedding.norm():.2f}")

    # Also save the difference embeddings (mutant - WT)
    wt_emb = embeddings["WT"]
    for allele in alleles:
        diff = embeddings[allele] - wt_emb
        torch.save(diff, ESM_DIR / f"{allele}_diff.pt")

    logger.info(f"All embeddings saved to {ESM_DIR}")
    return embeddings


def _generate_placeholder_embeddings(alleles: list) -> Dict[str, torch.Tensor]:
    """Generate random placeholder embeddings for development without ESM-2."""
    torch.manual_seed(42)
    embeddings = {"WT": torch.randn(1280)}
    torch.save(embeddings["WT"], ESM_DIR / "WT.pt")
    for a in alleles:
        embeddings[a] = torch.randn(1280)
        torch.save(embeddings[a], ESM_DIR / f"{a}.pt")
        diff = embeddings[a] - embeddings["WT"]
        torch.save(diff, ESM_DIR / f"{a}_diff.pt")
    logger.warning("Using PLACEHOLDER embeddings. Install fair-esm for real embeddings.")
    return embeddings


def load_esm2_embeddings(alleles: Optional[list] = None) -> Dict[str, torch.Tensor]:
    """Load pre-computed ESM-2 embeddings from disk.

    If *alleles* is None and embeddings exist for all known hotspots, loads
    all of them automatically — no need to enumerate manually.
    """
    if alleles is None:
        alleles = list(TP53_HOTSPOT_MUTATIONS.keys())

    embeddings = {}
    for name in ["WT"] + alleles:
        path = ESM_DIR / f"{name}.pt"
        if path.exists():
            embeddings[name] = torch.load(path, weights_only=True)
        else:
            logger.warning(f"Embedding not found for {name}. Run generate_esm2_embeddings() first.")

    return embeddings


def discover_available_embeddings() -> Dict[str, torch.Tensor]:
    """Auto-discover and load ALL embeddings present in ESM_DIR.

    Scans for ``<name>.pt`` files (excluding ``*_diff.pt``), loads them, and
    returns a dict keyed by allele name.  Unlike ``load_esm2_embeddings`` this
    does not require enumerating allele names in advance — useful when new
    allele embeddings have been added to disk without updating the hardcoded
    hotspot list.

    Returns:
        Dict[str, Tensor] mapping allele name → 1280-dim embedding.
    """
    if not ESM_DIR.exists():
        logger.warning(f"ESM2 embedding directory not found: {ESM_DIR}")
        return {}

    embeddings: Dict[str, torch.Tensor] = {}
    for pt_file in sorted(ESM_DIR.glob("*.pt")):
        name = pt_file.stem
        if name.endswith("_diff"):
            continue  # skip difference vectors; load main embeddings only
        try:
            embeddings[name] = torch.load(pt_file, weights_only=True)
        except Exception as e:
            logger.warning(f"Could not load {pt_file.name}: {e}")

    known = set(TP53_HOTSPOT_MUTATIONS) | {"WT"}
    present = set(embeddings)
    missing = known - present
    extra   = present - known

    if missing:
        logger.warning("Known hotspot alleles missing from disk: %s", sorted(missing))
    if extra:
        logger.info("Extra embeddings found (not in hotspot list): %s", sorted(extra))

    logger.info(
        "Discovered %d embeddings in %s: %s",
        len(embeddings), ESM_DIR.name, sorted(embeddings)
    )
    return embeddings


def validate_allele_coverage(embeddings: Optional[Dict[str, torch.Tensor]] = None) -> dict:
    """Check coverage of known TP53 hotspot alleles.

    Args:
        embeddings: Pre-loaded embedding dict. If None, calls
                    ``discover_available_embeddings()``.

    Returns:
        Report dict with keys ``present``, ``missing``, ``coverage_pct``,
        ``structural_coverage`` (contact vs conformational fractions).
    """
    if embeddings is None:
        embeddings = discover_available_embeddings()

    known   = set(TP53_HOTSPOT_MUTATIONS)
    present = set(embeddings) & known
    missing = known - present

    contact_present       = present & CONTACT_MUTANTS
    conformational_present = present & CONFORMATIONAL_MUTANTS

    report = {
        "present":            sorted(present),
        "missing":            sorted(missing),
        "coverage_pct":       100 * len(present) / len(known) if known else 0,
        "contact_coverage":   f"{len(contact_present)}/{len(CONTACT_MUTANTS)}",
        "conformational_coverage": f"{len(conformational_present)}/{len(CONFORMATIONAL_MUTANTS)}",
        "embedding_dim":      next(iter(embeddings.values())).shape[-1] if embeddings else None,
    }

    if missing:
        logger.warning("Allele coverage incomplete — missing: %s", report["missing"])
    else:
        logger.info("Full allele coverage: %d/7 hotspot alleles + WT", len(present))

    return report


def assign_alleles_to_cells(
    adata,
    mutation_df: pd.DataFrame,
    patient_col: str = "patient_id",
) -> None:
    """
    Assign TP53 allele labels to each cell in an AnnData object.

    Args:
        adata: AnnData with obs containing patient_col.
        mutation_df: DataFrame with columns [patient_id, tp53_allele].
        patient_col: Column name in adata.obs for patient ID.

    Adds adata.obs["tp53_allele"] with values like "R175H", "R273H", "WT", etc.
    """
    # Merge allele info onto cell metadata
    allele_map = mutation_df.set_index("patient_id")["tp53_allele"].to_dict()
    adata.obs["tp53_allele"] = adata.obs[patient_col].map(allele_map).fillna("unknown")

    # Report allele distribution
    counts = adata.obs["tp53_allele"].value_counts()
    logger.info("TP53 allele distribution across cells:")
    for allele, count in counts.items():
        logger.info(f"  {allele}: {count} cells")

    # Flag alleles with insufficient cells
    min_cells = 100
    for allele in counts.index:
        if allele != "unknown" and counts[allele] < min_cells:
            logger.warning(
                f"  WARNING: {allele} has only {counts[allele]} cells "
                f"(minimum recommended: {min_cells})"
            )


def get_structural_class(allele: str) -> str:
    """Return structural class of a TP53 allele."""
    if allele in CONTACT_MUTANTS:
        return "contact"
    elif allele in CONFORMATIONAL_MUTANTS:
        return "conformational"
    elif allele == "WT":
        return "wildtype"
    else:
        return "unknown"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    embeddings = generate_esm2_embeddings()
    print(f"\nGenerated embeddings for {len(embeddings)} alleles:")
    for name, emb in embeddings.items():
        print(f"  {name}: shape={emb.shape}, norm={emb.norm():.3f}")
