"""
TP53 allele biological class mapping.

Two functional classes of TP53 hotspot mutations, defined by mechanism:
  Class 0 — WT (wildtype control)
  Class 1 — Structural mutants: destabilize protein fold (R175H, Y220C)
  Class 2 — DNA-contact mutants: protein folds but cannot bind DNA
              (R248W, R248Q, R273H, R282W, G245S)

Cell counts in BRCA Wu2021 (from HPC logs 2026-05-18):
  WT: 13,875  |  structural: ~4,142  |  dna_contact: ~6,472

References:
  Petitjean A. et al. (2007) Hum Mutat 28(6):622-629 -- TP53 mutation database
  Cho Y. et al. (1994) Science 265:346-355 -- structural basis of TP53 mutations
"""

__all__ = [
    "ALLELE_TO_CLASS",
    "CLASS_NAMES",
    "CLASS_DESCRIPTIONS",
    "N_ALLELE_CLASSES",
    "allele_label_to_class_id",
    "get_class_name",
]

# Canonical mapping -- all known hotspot alleles
ALLELE_TO_CLASS: dict[str, int] = {
    # Class 0: wildtype
    "WT": 0,
    "wt": 0,
    "wildtype": 0,
    "unknown": 0,  # fallback for unannotated cells
    # Class 1: structural (fold-destabilizing)
    "R175H": 1,
    "Y220C": 1,
    # Class 2: DNA-contact (DNA-binding deficient)
    "R248W": 2,
    "R248Q": 2,
    "R273H": 2,
    "R282W": 2,
    "G245S": 2,
    "R249S": 2,
    "R245W": 2,
}

CLASS_NAMES: list[str] = ["WT", "structural", "dna_contact"]

CLASS_DESCRIPTIONS: list[str] = [
    "Wildtype TP53 -- functional p53 tetramer with normal DNA binding",
    "Structural mutants -- destabilize protein fold (R175H, Y220C). "
    "beta-sandwich core disrupted; protein cannot maintain active conformation.",
    "DNA-contact mutants -- protein folds correctly but cannot grip DNA "
    "(R248W/Q, R273H, R282W, G245S). L3-loop and S10-strand contact residues lost.",
]

N_ALLELE_CLASSES: int = len(CLASS_NAMES)


def allele_label_to_class_id(allele: str) -> int:
    """Map allele string to class integer. Unknown alleles -> class 0 (WT)."""
    return ALLELE_TO_CLASS.get(allele, ALLELE_TO_CLASS.get(allele.strip(), 0))


def get_class_name(class_id: int) -> str:
    """Return human-readable name for a class integer."""
    if 0 <= class_id < N_ALLELE_CLASSES:
        return CLASS_NAMES[class_id]
    raise ValueError(f"class_id {class_id} out of range [0, {N_ALLELE_CLASSES})")
