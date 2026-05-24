import torch
import pytest
from tsgnn.model.allele_so4 import AlleleSO4Embedding
from tsgnn.data.allele_classes import N_ALLELE_CLASSES


STALK_DIM = 4


@pytest.fixture
def embedding():
    return AlleleSO4Embedding(n_allele_classes=N_ALLELE_CLASSES, stalk_dim=STALK_DIM)


def test_output_shape(embedding):
    class_id = torch.tensor(0)
    R = embedding(class_id)
    assert R.shape == (STALK_DIM, STALK_DIM), f"Expected ({STALK_DIM},{STALK_DIM}), got {R.shape}"


def test_output_is_orthogonal(embedding):
    """R must satisfy R^T R ≈ I (orthogonal matrix)."""
    for cls in range(N_ALLELE_CLASSES):
        R = embedding(torch.tensor(cls))
        eye_approx = R.T @ R
        I = torch.eye(STALK_DIM)
        assert torch.allclose(eye_approx, I, atol=1e-5), \
            f"Class {cls} rotation not orthogonal: ||R^T R - I||={(eye_approx - I).norm():.6f}"


def test_output_has_det_one(embedding):
    """det(R) must be +1 for SO(d) (not just O(d))."""
    for cls in range(N_ALLELE_CLASSES):
        R = embedding(torch.tensor(cls))
        det = torch.linalg.det(R)
        assert abs(det.item() - 1.0) < 1e-4, \
            f"Class {cls} det={det.item():.6f}, expected 1.0"


def test_different_classes_produce_different_rotations(embedding):
    """After any gradient step, classes should learn different rotations."""
    with torch.no_grad():
        embedding.skew_params.weight[0] = torch.zeros(embedding.n_skew)
        embedding.skew_params.weight[1] = torch.ones(embedding.n_skew) * 0.5
        embedding.skew_params.weight[2] = torch.ones(embedding.n_skew) * -0.5

    R0 = embedding(torch.tensor(0))
    R1 = embedding(torch.tensor(1))
    R2 = embedding(torch.tensor(2))

    assert not torch.allclose(R0, R1, atol=1e-3), "Class 0 and 1 should differ"
    assert not torch.allclose(R1, R2, atol=1e-3), "Class 1 and 2 should differ"


def test_wt_class_initializes_near_identity(embedding):
    """WT (class 0) starts near identity -- small random init."""
    R_wt = embedding(torch.tensor(0))
    I = torch.eye(STALK_DIM)
    diff = (R_wt - I).norm()
    assert diff < 0.5, f"WT rotation too far from identity at init: ||R-I||={diff:.4f}"


def test_apply_to_maps(embedding):
    """F_allele = F_base @ R -- shape preserved, F_allele in O(d) if F_base in O(d)."""
    E = 100
    d = STALK_DIM
    # Build orthogonal F_base
    F_base = torch.zeros(E, 2, d, d)
    for e in range(E):
        for s in range(2):
            F_base[e, s] = torch.linalg.qr(torch.randn(d, d))[0]

    R = embedding(torch.tensor(1))  # structural class
    F_allele = torch.einsum('esij,jk->esik', F_base, R)  # (E, 2, d, d)

    assert F_allele.shape == (E, 2, d, d)

    # Check F_allele is still O(d)
    sample = F_allele[0, 0]
    eye_approx = sample.T @ sample
    I = torch.eye(d)
    assert torch.allclose(eye_approx, I, atol=1e-5)


def test_gradients_flow(embedding):
    """Gradients must flow through the Cayley/exp map back to skew_params."""
    class_id = torch.tensor(1)
    R = embedding(class_id)
    loss = R.sum()
    loss.backward()
    grad = embedding.skew_params.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all(), "Non-finite gradients in skew_params"
    assert grad[1].abs().sum() > 0, "Zero gradient for class 1 -- no signal"


def test_n_parameters():
    """Total parameters = n_classes * n_skew (18 for 3 classes, d=4)."""
    emb = AlleleSO4Embedding(n_allele_classes=3, stalk_dim=4)
    n_params = sum(p.numel() for p in emb.parameters())
    assert n_params == 3 * 6, f"Expected 18 params, got {n_params}"
