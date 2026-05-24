import torch
import pytest
from tsgnn.model.tsgnn import TSGNN


N, E, d, K = 30, 100, 4, 5
INPUT_DIM = 1


@pytest.fixture
def edge_index():
    return torch.randint(0, N, (2, E))


@pytest.fixture
def model(edge_index):
    return TSGNN(
        num_nodes=N,
        num_edges=E,
        stalk_dim=d,
        input_dim=INPUT_DIM,
        n_allele_classes=3,
        edge_index=edge_index,
    )


def test_forward_accepts_allele_class(model):
    x = torch.randn(K, N, INPUT_DIM)
    allele_class = torch.tensor(1)  # structural
    preds, maps, laplacians = model(x, allele_class)
    assert len(preds) == K
    assert preds[0].shape == (N, INPUT_DIM)


def test_different_allele_classes_produce_different_predictions(model):
    x = torch.randn(K, N, INPUT_DIM)
    with torch.no_grad():
        model.allele_so4.skew_params.weight[0].zero_()
        model.allele_so4.skew_params.weight[1] = torch.ones(d * (d - 1) // 2) * 0.5
        model.allele_so4.skew_params.weight[2] = torch.ones(d * (d - 1) // 2) * -0.5

    preds_wt, _, _ = model(x, torch.tensor(0))
    preds_struct, _, _ = model(x, torch.tensor(1))

    for t in range(K):
        assert not torch.allclose(preds_wt[t], preds_struct[t], atol=1e-4), \
            f"Step {t}: WT and structural predictions are identical"


def test_restriction_maps_are_orthogonal(model):
    x = torch.randn(K, N, INPUT_DIM)
    _, maps, _ = model(x, torch.tensor(0))
    for t, M in enumerate(maps):
        F_src = M[:, 0]  # (E, d, d)
        eye_approx = torch.bmm(F_src.transpose(1, 2), F_src)
        I = torch.eye(d).unsqueeze(0).expand(E, -1, -1)
        max_err = (eye_approx - I).abs().max().item()
        assert max_err < 0.1, f"Step {t}: maps not orthogonal (max err={max_err:.4f})"


def test_n_parameters_is_minimal(model):
    """SO4 embedding should be 18 params (3 classes x 6 so(4) dims)."""
    so4_params = sum(p.numel() for p in model.allele_so4.parameters())
    assert so4_params == 3 * (d * (d - 1) // 2), \
        f"Expected 18 SO4 params, got {so4_params}"


def test_no_esm_input_required(edge_index):
    """Model must not require ESM-2 input -- allele_class is sufficient."""
    model = TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=d,
        input_dim=INPUT_DIM, n_allele_classes=3, edge_index=edge_index,
    )
    x = torch.randn(K, N, INPUT_DIM)
    preds, _, _ = model(x, torch.tensor(2))
    assert len(preds) == K


def test_allele_divergence_nonzero_after_init(model):
    """Structural vs WT predictions must differ even at initialization."""
    from tsgnn.evaluation.geometric_metrics import allele_divergence
    x = torch.randn(K, N, INPUT_DIM)

    with torch.no_grad():
        model.allele_so4.skew_params.weight[1] = torch.ones(d * (d - 1) // 2) * 0.3

    preds_wt = torch.stack(model(x, torch.tensor(0))[0]).mean(dim=1).flatten()
    preds_struct = torch.stack(model(x, torch.tensor(1))[0]).mean(dim=1).flatten()
    div = allele_divergence(preds_struct, preds_wt)
    assert div > 1e-4, f"allele_divergence={div:.6f} -- SO4 not producing signal"
