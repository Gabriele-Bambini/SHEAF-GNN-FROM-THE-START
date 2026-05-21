"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
import torch

from tsgnn.evaluation.metrics import (
    cross_cancer_zero_shot,
    allele_edge_enrichment,
    chromatin_remodeler_concordance,
    depmap_dependency_correlation,
    rewiring_distinguishability,
    extract_differential_edges,
    compute_hub_influence_scores,
)


# ── 1. Cross-Cancer Zero-Shot ────────────────────────────────────────────

def test_zero_shot_perfect_correlation():
    """Identical predicted and observed edges → r ≈ 1.0."""
    edges = torch.randn(100)
    result = cross_cancer_zero_shot(edges, edges)
    assert result["pearson_r"] > 0.99
    assert result["p_value"] < 0.01
    assert result["interpretation"] == "strong_generalization"


def test_zero_shot_no_correlation():
    """Orthogonal predictions → low r, tissue_dominant interpretation."""
    torch.manual_seed(0)
    pred = torch.randn(500)
    obs = torch.randn(500)
    result = cross_cancer_zero_shot(pred, obs)
    # With random data, |r| should be small
    assert abs(result["pearson_r"]) < 0.3


def test_zero_shot_handles_nan():
    """Should handle NaN/Inf values gracefully."""
    pred = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0])
    obs = torch.tensor([1.0, 2.0, float("inf"), 4.0, 5.0])
    result = cross_cancer_zero_shot(pred, obs)
    assert "pearson_r" in result


def test_zero_shot_insufficient_data():
    """Should return insufficient_data for < 3 valid points."""
    pred = torch.tensor([1.0, float("nan")])
    obs = torch.tensor([float("nan"), 2.0])
    result = cross_cancer_zero_shot(pred, obs)
    assert result["interpretation"] == "insufficient_data"


# ── 2. Allele Edge Enrichment ────────────────────────────────────────────

def test_enrichment_perfect_overlap():
    """When top edges ⊂ validated targets, odds ratio should be high."""
    top = [(0, 1), (1, 2), (2, 3)]
    validated = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    result = allele_edge_enrichment(top, validated, total_possible_edges=100)
    assert result["overlap_count"] == 3
    assert result["odds_ratio"] > 1.0


def test_enrichment_no_overlap():
    """When no overlap, odds ratio should be ≤ 1."""
    top = [(10, 11), (12, 13)]
    validated = [(0, 1), (1, 2)]
    result = allele_edge_enrichment(top, validated, total_possible_edges=100)
    assert result["overlap_count"] == 0


def test_enrichment_direction_concordance():
    """Direction concordance should be computed correctly when provided."""
    edges = [(0, 1), (1, 2)]
    validated = [(0, 1), (1, 2)]
    pred_dir = {(0, 1): 1.0, (1, 2): -1.0}
    known_dir = {(0, 1): 1.0, (1, 2): -1.0}
    result = allele_edge_enrichment(
        edges, validated, 100,
        predicted_directions=pred_dir, known_directions=known_dir,
    )
    assert result["direction_concordance"] == 1.0


# ── 3. Chromatin Remodeler Concordance ───────────────────────────────────

def test_chip_seq_jaccard():
    """Jaccard index should be computed correctly."""
    hw = [(0, 1), (1, 2), (2, 3)]
    chip = [(1, 2), (2, 3), (3, 4)]
    result = chromatin_remodeler_concordance(hw, chip, 100)
    # Intersection = {(1,2), (2,3)} = 2, Union = 4
    assert result["jaccard_index"] == pytest.approx(0.5)
    assert result["overlap_count"] == 2


# ── 4. DepMap Dependency Correlation ─────────────────────────────────────

def test_depmap_perfect_correlation():
    """Identical scores → rho ≈ 1."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = depmap_dependency_correlation(scores, scores)
    assert result["spearman_rho"] > 0.99


def test_depmap_insufficient_data():
    """Should handle < 3 valid data points."""
    scores = np.array([1.0, np.nan])
    result = depmap_dependency_correlation(scores, scores)
    assert result["spearman_rho"] == 0.0


# ── 5. Rewiring Distinguishability ───────────────────────────────────────

def test_distinguishability_identical_alleles():
    """Identical alleles → high p-value (not distinguishable).

    Theoretical: if delta_L_a == delta_L_b, the observed statistic should
    be near 0, and the permutation distribution should produce similar values,
    giving p ≈ 1.0.
    """
    K, Nd = 5, 20
    laps = [torch.randn(Nd, Nd) for _ in range(K)]
    result = rewiring_distinguishability(laps, laps, n_permutations=100)
    # Observed stat should be 0 (identical inputs)
    assert result["observed_stat"] < 1e-5


def test_distinguishability_different_alleles():
    """Highly different alleles → low p-value (distinguishable).

    With laps_b scaled 10x larger than laps_a, the observed L2 distance
    between mean Laplacians should be far in the permutation distribution.
    Using 500 permutations for stable p-value estimation.
    """
    K, Nd = 5, 20
    torch.manual_seed(42)
    laps_a = [torch.randn(Nd, Nd) for _ in range(K)]
    laps_b = [torch.randn(Nd, Nd) * 10 for _ in range(K)]  # 10x scale
    result = rewiring_distinguishability(laps_a, laps_b, n_permutations=500)
    assert result["p_value"] < 0.2  # Should be clearly distinguishable (10x scale diff)
    assert result["effect_size"] > 1.0


# ── Utility Functions ────────────────────────────────────────────────────

def test_extract_differential_edges_shape():
    """Should return top_k edges with correct count."""
    K, N, d = 5, 10, 2
    Nd = N * d
    laps_a = [torch.randn(Nd, Nd) for _ in range(K)]
    laps_b = [torch.randn(Nd, Nd) for _ in range(K)]
    edges, weights = extract_differential_edges(laps_a, laps_b, d, top_k=5)
    assert len(edges) == 5
    assert weights.shape == (5,)


def test_hub_influence_scores_shape():
    """Hub scores should have shape (N,) with positive values."""
    K, N, d = 5, 10, 2
    Nd = N * d
    laps = [torch.randn(Nd, Nd) for _ in range(K)]
    scores = compute_hub_influence_scores(laps, d)
    assert scores.shape == (N,)
    assert (scores >= 0).all()
