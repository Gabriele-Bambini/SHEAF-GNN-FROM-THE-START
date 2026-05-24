import os
import requests
from pathlib import Path

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Done: {dest.stat().st_size / 1e6:.1f} MB")

base_dir = Path(os.getcwd()) / "data" / "raw" / "brca"
os.makedirs(base_dir, exist_ok=True)

# GSE176078
gse176_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"
dest176 = base_dir / "GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"
if not dest176.exists():
    try: download_file(gse176_url, dest176)
    except Exception as e: print(f"Error GSE176078: {e}")

# GSE158508 
gse158_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE158nnn/GSE158508/suppl/GSE158508_normalized_counts.txt.gz"
dest158 = base_dir / "GSE158508_normalized_counts.txt.gz"
if not dest158.exists():
    try: download_file(gse158_url, dest158)
    except Exception as e: print(f"Error GSE158508: {e}")
