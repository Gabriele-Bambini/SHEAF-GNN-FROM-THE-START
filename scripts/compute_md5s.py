import hashlib
from pathlib import Path

FILES_TO_CHECK = {
    "GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz": Path("data/raw/brca/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"),
    "9606.protein.links.v12.0.txt.gz":             Path("data/external/string/9606.protein.links.v12.0.txt.gz"),
    "JASPAR2024_CORE_vertebrates.txt":             Path("data/external/jaspar/JASPAR2024_CORE_vertebrates.txt"),
    "tcga_brca_tp53_mutations.csv":                Path("data/external/tcga/tcga_brca_tp53_mutations.csv"),
}

for name, path in FILES_TO_CHECK.items():
    if path.exists():
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        print(f'    "{name}": "{h.hexdigest()}",')
    else:
        print(f'    # MANCANTE: {name}')
