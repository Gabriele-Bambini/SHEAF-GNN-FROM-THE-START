#!/bin/bash
# Download Izar 2020 HGSOC scRNA-seq (GSE146026) on HPC.
# Files: GSE146026_Izar_HGSOC_ascites_10x_log.tsv.gz (98MB)
#        GSE146026_Izar_HGSOC_ascites_SS2_log.tsv.gz (54MB)
#
# Usage: bash hpc/download_ovarian_data.sh

set -euo pipefail

DEST_DIR="${HOME}/ts-gnn/data/raw/ovarian"
mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE146nnn/GSE146026/suppl"

# Download 10x (primary — more cells)
F10X="GSE146026_Izar_HGSOC_ascites_10x_log.tsv.gz"
if [[ ! -f "${F10X%.gz}" && ! -f "${F10X}" ]]; then
    echo "Downloading ${F10X} (98MB)..."
    wget -q --show-progress "${BASE}/${F10X}"
else
    echo "${F10X} already present"
fi

# Download SS2 (secondary — different tech)
FSS2="GSE146026_Izar_HGSOC_ascites_SS2_log.tsv.gz"
if [[ ! -f "${FSS2%.gz}" && ! -f "${FSS2}" ]]; then
    echo "Downloading ${FSS2} (54MB)..."
    wget -q --show-progress "${BASE}/${FSS2}"
else
    echo "${FSS2} already present"
fi

echo "=== Download complete in ${DEST_DIR} ==="
ls -lh "${DEST_DIR}"
