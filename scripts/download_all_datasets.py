#!/usr/bin/env python3
"""
TS-GNN Dataset Downloader — run on Colab with Drive mounted.

Usage on Colab:
    !python scripts/download_all_datasets.py

Downloads everything into data/ relative to the project root.
All paths align with what src/tsgnn/data/*.py expect.
"""

import logging
from pathlib import Path

from tsgnn.data import DATA_DIR, RAW_DIR, EXTERNAL_DIR
import tsgnn.data.download as download_mod
from tsgnn.data.download import (
    download_brca_scrna,
    download_tcga_brca,
    download_jaspar,
    download_string,
    download_regnetwork,
    download_p53_targets,
    download_depmap,
    download_encode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download")

log.info(f"Using download module from: {download_mod.__file__}")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIRS = [
    RAW_DIR / "brca" / "GSE176078",
    RAW_DIR / "brca" / "GSE158508",
    RAW_DIR / "tcga",
    EXTERNAL_DIR / "jaspar",
    EXTERNAL_DIR / "string",
    EXTERNAL_DIR / "regnetwork",
    EXTERNAL_DIR / "targetgenereg",
    EXTERNAL_DIR / "depmap",
    EXTERNAL_DIR / "encode",
    EXTERNAL_DIR / "esm2_embeddings",
    DATA_DIR / "processed",
]


def mkdirs():
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Validation: check all expected files exist
# ═══════════════════════════════════════════════════════════════════════════
def validate():
    log.info("=" * 60)
    log.info("VALIDATION")
    log.info("=" * 60)

    checks = [
        (RAW_DIR / "brca" / "GSE176078" / "GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz",
         "BRCA scRNA-seq primary cohort", True),
        (RAW_DIR / "brca" / "GSE158508" / "GSE158508_normalized_counts.txt.gz",
         "BRCA scRNA-seq supplementary cohort", False),
        (RAW_DIR / "tcga" / "tcga_brca_tp53_mutations.csv",
         "TCGA BRCA TP53 mutations", True),
        (EXTERNAL_DIR / "jaspar" / "JASPAR2024_CORE_vertebrates.txt",
         "JASPAR 2024 TF motifs", True),
        (EXTERNAL_DIR / "string" / "9606.protein.links.v12.0.txt.gz",
         "STRING v12 PPI", True),
        (EXTERNAL_DIR / "string" / "9606.protein.aliases.v12.0.txt.gz",
         "STRING v12 aliases", True),
        (EXTERNAL_DIR / "regnetwork" / "human_regulatory.txt",
         "RegNetwork", True),
        (EXTERNAL_DIR / "targetgenereg" / "p53_targets_benchmark.csv",
         "Fischer 2017 p53 targets", True),
        (EXTERNAL_DIR / "encode" / "TP53_MCF7_peaks.bed.gz",
         "ENCODE BRCA ChIP-seq", False),
    ]

    ok = 0
    fail = 0
    for path, desc, critical in checks:
        if path.exists() and path.stat().st_size > 50:
            log.info(f"  OK   {desc}: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
            ok += 1
        else:
            level = "FAIL" if critical else "WARN"
            log.warning(f"  {level} {desc}: MISSING — {path}")
            fail += 1

    log.info(f"\n  {ok} OK, {fail} missing")

    # Check TCGA TP53 alleles
    tcga = RAW_DIR / "tcga" / "tcga_brca_tp53_mutations.csv"
    if tcga.exists():
        import csv
        with open(tcga) as f:
            reader = csv.DictReader(f)
            alleles = {}
            for row in reader:
                a = row.get("tp53_allele", row.get("protein_change", ""))
                if a:
                    alleles[a] = alleles.get(a, 0) + 1
        if alleles:
            log.info(f"\n  TCGA BRCA TP53 alleles found ({len(alleles)} unique):")
            for a, count in sorted(alleles.items(), key=lambda x: -x[1])[:15]:
                log.info(f"    {a}: {count} patients")

    log.info("\n" + "=" * 60)
    log.info("DONE")
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("TS-GNN Dataset Downloader")
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Data dir: {DATA_DIR}")
    log.info("=" * 60)

    log.info("[STEP 0] Ensuring directories...")
    mkdirs()

    log.info("[STEP 1] Downloading BRCA scRNA-seq datasets...")
    download_brca_scrna()

    log.info("[STEP 2] Downloading TCGA BRCA mutations...")
    download_tcga_brca()

    log.info("[STEP 3] Downloading JASPAR motifs...")
    download_jaspar()

    log.info("[STEP 4] Downloading STRING PPI network...")
    download_string()

    log.info("[STEP 5] Downloading RegNetwork...")
    download_regnetwork()

    log.info("[STEP 6] Downloading Fischer p53 targets...")
    download_p53_targets()

    log.info("[STEP 7] Downloading DepMap CRISPR effects...")
    download_depmap()

    log.info("[STEP 8] Downloading ENCODE ChIP-seq peaks...")
    download_encode()

    log.info("[STEP 9] Running final validation check...")
    validate()

    log.info("=" * 60)
    log.info("Process finished. If any warnings were shown, check the specific logs.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
