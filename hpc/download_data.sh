#!/bin/bash
# Download all raw datasets directly on HPC from GEO/EBI.
# Run once after first deploy: bash hpc/download_data.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ts-gnn}"
DATA="$REPO_DIR/data/raw"
mkdir -p "$DATA/brca/GSE176078" "$DATA/brca/GSE158508" \
         "$DATA/brca/BIOKEY_E-MTAB-8107" \
         "$DATA/crc" "$DATA/metabric" "$DATA/tcga"

echo "=== Downloading datasets directly on HPC ==="

# ── GSE176078 (Wu et al. 2021 BRCA, ~533 MB) ─────────────────────────────
F="$DATA/brca/GSE176078/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"
if [ ! -f "$F" ]; then
  echo "[1/6] GSE176078 BRCA scRNA-seq..."
  wget -q --show-progress -O "$F" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"
else echo "[1/6] GSE176078 already present, skip."; fi

# ── GSE176078 metadata ────────────────────────────────────────────────────
F2="$DATA/brca/GSE176078/GSE176078_metadata.csv"
if [ ! -f "$F2" ]; then
  wget -q -O "$F2" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/GSE176078_metadata.csv" \
    2>/dev/null || echo "  metadata optional, skip"
fi

# ── GSE158508 (normalized counts, ~4.4 MB gz) ────────────────────────────
F="$DATA/brca/GSE158508/GSE158508_normalized_counts.txt"
FGZ="$DATA/brca/GSE158508/GSE158508_normalized_counts.txt.gz"
if [ ! -f "$F" ] && [ ! -f "$FGZ" ]; then
  echo "[2/6] GSE158508 normalized counts..."
  wget -q --show-progress -O "$FGZ" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE158nnn/GSE158508/suppl/GSE158508_normalized_counts.txt.gz"
  gunzip "$FGZ"
else echo "[2/6] GSE158508 already present, skip."; fi

# ── BIOKEY E-MTAB-8107 (EBI ArrayExpress) ────────────────────────────────
BIOKEY="$DATA/brca/BIOKEY_E-MTAB-8107"
if [ ! -f "$BIOKEY/E-MTAB-8107.sdrf.txt" ]; then
  echo "[3/6] BIOKEY E-MTAB-8107 metadata..."
  wget -q -O "$BIOKEY/E-MTAB-8107.sdrf.txt" \
    "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-8107/E-MTAB-8107.sdrf.txt"
  wget -q -O "$BIOKEY/E-MTAB-8107.idf.txt" \
    "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-8107/E-MTAB-8107.idf.txt"
  echo "  BIOKEY counts must be downloaded manually from EBI (requires login)."
  echo "  See: https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-8107/files/"
else echo "[3/6] BIOKEY already present, skip."; fi

# ── CRC GSE178341 (Pelka et al. 2021, ~1.1 GB) ───────────────────────────
F="$DATA/crc/GSE178341_crc10x_full_c295v4_submit.h5"
if [ ! -f "$F" ]; then
  echo "[4/6] CRC GSE178341 HDF5..."
  wget -q --show-progress -O "$F" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178341/suppl/GSE178341_crc10x_full_c295v4_submit.h5"
else echo "[4/6] CRC already present, skip."; fi

# ── METABRIC (cBioPortal, ~700 MB) ───────────────────────────────────────
MET="$DATA/metabric"
if [ ! -f "$MET/data_mrna_illumina_microarray.txt" ]; then
  echo "[5/6] METABRIC from cBioPortal..."
  TMP=$(mktemp -d)
  wget -q --show-progress -O "$TMP/metabric.tar.gz" \
    "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz"
  tar -xzf "$TMP/metabric.tar.gz" -C "$TMP"
  cp "$TMP"/brca_metabric/data_clinical_patient.txt "$MET/" 2>/dev/null || true
  cp "$TMP"/brca_metabric/data_clinical_sample.txt  "$MET/" 2>/dev/null || true
  cp "$TMP"/brca_metabric/data_mutations.txt        "$MET/" 2>/dev/null || true
  cp "$TMP"/brca_metabric/data_mrna_illumina_microarray.txt "$MET/" 2>/dev/null || true
  rm -rf "$TMP"
else echo "[5/6] METABRIC already present, skip."; fi

# ── TCGA TP53 mutations ───────────────────────────────────────────────────
F="$DATA/tcga/tcga_brca_tp53_mutations.csv"
if [ ! -f "$F" ]; then
  echo "[6/6] TCGA TP53 mutations (from GDC)..."
  echo "  Manual download required — export from GDC portal."
  echo "  https://portal.gdc.cancer.gov/exploration?filters=TP53"
else echo "[6/6] TCGA already present, skip."; fi

echo ""
echo "=== Download complete. Verify with: ==="
echo "  python hpc/check_data.py --data-dir data --config configs/nature.yaml"
