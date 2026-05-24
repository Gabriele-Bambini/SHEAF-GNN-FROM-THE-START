"""Test end-to-end su dati BRCA reali (skip se file non presenti)."""
import pytest
from pathlib import Path
from tsgnn.data import RAW_DIR

BRCA_PATH = RAW_DIR / "brca" / "GSE176078"

@pytest.mark.slow
@pytest.mark.skipif(not BRCA_PATH.exists(), reason="Dati BRCA non scaricati")
def test_full_pipeline_brca():
    from tsgnn.data.brca_loader import load_brca_data, build_brca_temporal_sequences
    from tsgnn.data.grn_construction import construct_base_grn

    adata = load_brca_data()
    assert adata.n_obs > 0
    assert "tp53_allele" in adata.obs.columns

    # Un mock edge_index per il test, dato che construct_base_grn richiede tf_list ecc.
    import torch
    edge_index = torch.randint(0, adata.n_vars, (2, 100))

    sequences = build_brca_temporal_sequences(adata, K=10, split="train", edge_index=edge_index, gene_list=list(adata.var_names))
    assert len(sequences) > 0

    for allele_name, seq in sequences.items():
        # seq is TemporalGraphSequence — use .node_features_sequence()
        nf = seq.node_features_sequence()
        assert nf.shape[0] == 10  # K bins
