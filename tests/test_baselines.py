"""Unit tests for baseline models (interface uniformity and correctness)."""

import pytest
import torch

from tsgnn.model.baselines import (
    EvolveGCN,
    TemporalGAT,
    create_tsgnn_no_allele,
    create_baseline,
    BASELINE_REGISTRY,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

N, E, K, INPUT_DIM, HIDDEN, ESM_DIM = 20, 60, 5, 20, 16, 1280


@pytest.fixture
def edge_index():
    return torch.randint(0, N, (2, E))


@pytest.fixture
def node_features_seq():
    return torch.randn(K, N, INPUT_DIM)


@pytest.fixture
def allele_emb():
    return torch.randn(ESM_DIM)


# ── EvolveGCN Tests ──────────────────────────────────────────────────────

def test_evolvegcn_forward_shape(edge_index, node_features_seq):
    model = EvolveGCN(
        num_nodes=N, input_dim=INPUT_DIM, hidden_dim=HIDDEN, edge_index=edge_index
    )
    preds, maps, laps = model(node_features_seq)
    assert len(preds) == K
    assert preds[0].shape == (N, INPUT_DIM)
    assert len(maps) == K
    assert len(laps) == K


def test_evolvegcn_backward(edge_index, node_features_seq):
    model = EvolveGCN(
        num_nodes=N, input_dim=INPUT_DIM, hidden_dim=HIDDEN, edge_index=edge_index
    )
    preds, _, _ = model(node_features_seq)
    loss = sum(p.sum() for p in preds)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ── Temporal GAT Tests ───────────────────────────────────────────────────

def test_temporal_gat_forward_shape(edge_index, node_features_seq, allele_emb):
    model = TemporalGAT(
        num_nodes=N, input_dim=INPUT_DIM, hidden_dim=HIDDEN, edge_index=edge_index
    )
    preds, maps, laps = model(node_features_seq, allele_emb)
    assert len(preds) == K
    assert preds[0].shape == (N, INPUT_DIM)


def test_temporal_gat_backward(edge_index, node_features_seq, allele_emb):
    model = TemporalGAT(
        num_nodes=N, input_dim=INPUT_DIM, hidden_dim=HIDDEN, edge_index=edge_index
    )
    preds, _, _ = model(node_features_seq, allele_emb)
    loss = sum(p.sum() for p in preds)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ── TS-GNN No-Allele (Ablation) Tests ────────────────────────────────────

def test_tsgnn_no_allele_forward(edge_index, node_features_seq, allele_emb):
    model = create_tsgnn_no_allele(
        num_nodes=N, num_edges=E, stalk_dim=4, input_dim=INPUT_DIM,
        edge_index=edge_index,
    )
    preds, maps, laps = model(node_features_seq, allele_emb)
    assert len(preds) == K
    assert preds[0].shape == (N, INPUT_DIM)


# ── Interface Uniformity Tests ───────────────────────────────────────────

def test_all_baselines_return_three_lists(edge_index, node_features_seq, allele_emb):
    """All baselines must return (predictions, maps, laplacians) — 3 lists of len K.

    Theoretical justification: this uniform interface enables fair comparison
    via identical evaluation code. Without it the benchmark runner would need
    per-baseline special-casing, introducing bugs.
    """
    models = {
        "evolvegcn": EvolveGCN(
            num_nodes=N, input_dim=INPUT_DIM, hidden_dim=HIDDEN, edge_index=edge_index
        ),
        "temporal_gat": TemporalGAT(
            num_nodes=N, input_dim=INPUT_DIM, hidden_dim=HIDDEN, edge_index=edge_index
        ),
        "tsgnn_no_allele": create_tsgnn_no_allele(
            num_nodes=N, num_edges=E, stalk_dim=4, input_dim=INPUT_DIM,
            edge_index=edge_index,
        ),
    }
    for name, model in models.items():
        preds, maps, laps = model(node_features_seq, allele_emb)
        assert isinstance(preds, list), f"{name}: predictions not a list"
        assert isinstance(maps, list), f"{name}: maps not a list"
        assert isinstance(laps, list), f"{name}: laplacians not a list"
        assert len(preds) == K, f"{name}: wrong prediction count"
        assert preds[0].shape == (N, INPUT_DIM), f"{name}: wrong prediction shape"


# ── Registry Tests ───────────────────────────────────────────────────────

def test_registry_has_all_baselines():
    """Registry should contain all planned baselines (incl. Nature-phase additions)."""
    expected = {"celloracle", "dictys", "evolvegcn", "tsgnn_no_allele", "temporal_gat",
                "mlp", "gcn_static", "dysat"}
    assert expected.issubset(set(BASELINE_REGISTRY.keys()))


def test_create_baseline_unknown_raises():
    with pytest.raises(ValueError, match="Unknown baseline"):
        create_baseline("nonexistent_model")
