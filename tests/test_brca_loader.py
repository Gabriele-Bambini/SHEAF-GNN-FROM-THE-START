"""Test BRCA data loader con file mock."""
import pytest
import os
from pathlib import Path

def test_tcga_allele_mapping(tmp_path):
    """Verifica mapping proteinChange → allele name."""
    import pandas as pd
    from tsgnn.data.brca_loader import load_tcga_brca_tp53
    import tsgnn.data.brca_loader as bl

    # Crea CSV mock
    df = pd.DataFrame({
        "sampleId": ["TCGA-A1-A001", "TCGA-A1-A002", "TCGA-A1-A003"],
        "proteinChange": ["p.R175H", "p.G245S", "p.V143A"],
    })
    
    # Sposta il file nella posizione attesa
    tcga_dir = tmp_path / "tcga"
    tcga_dir.mkdir()
    csv_path = tcga_dir / "tcga_brca_tp53_mutations.csv"
    df.to_csv(csv_path, index=False)

    # Monkey-patch RAW_DIR
    original_raw_dir = getattr(bl, "RAW_DIR", None)
    bl.RAW_DIR = tmp_path
    
    try:
        result = load_tcga_brca_tp53(tcga_dir)
        assert "R175H" in result["tp53_allele"].values
        assert "G245S" in result["tp53_allele"].values
        assert "other_mutation" in result["tp53_allele"].values  # V143A non modellato
    finally:
        if original_raw_dir is not None:
            bl.RAW_DIR = original_raw_dir
