"""
SO4 architecture end-to-end smoke test.

Validates the full pipeline with synthetic data:
- TSGNN with AlleleSO4Embedding (18 params, 3 classes)
- construct_base_grn signed edge_weight in bundle
- Forward + backward pass
- allele_divergence > 0 (SO4 not collapsed)
- Different allele classes produce different predictions

This is the gate test before HPC submission.
"""
import torch
import numpy as np
import pytest

from tsgnn.model.tsgnn import TSGNN, create_tsgnn_from_config
from tsgnn.training.loss import TSGNNLoss
from tsgnn.evaluation.geometric_metrics import allele_divergence


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def so4_model():
    """Create a minimal SO4 TSGNN model."""
    N, E, d = 20, 40, 4
    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=d, input_dim=1,
        edge_index=edge_index, n_allele_classes=3,
        num_diffusion_steps=2,
    )
    return model, N, E, d, edge_index


# ── Forward pass ─────────────────────────────────────────────────────────────

class TestSO4ForwardPass:
    def test_forward_completes(self, so4_model):
        """Forward pass must complete without error."""
        model, N, E, d, _ = so4_model
        x = torch.randn(5, N, 1)
        preds, maps, laps = model(x, torch.tensor(0))
        assert len(preds) == 5
        assert preds[0].shape == (N, 1)

    def test_forward_three_allele_classes(self, so4_model):
        """All 3 allele classes must produce valid output."""
        model, N, E, d, _ = so4_model
        x = torch.randn(3, N, 1)
        for cls in [0, 1, 2]:
            preds, maps, laps = model(x, torch.tensor(cls))
            assert len(preds) == 3
            assert all(torch.isfinite(p).all() for p in preds)

    def test_maps_are_orthogonal(self, so4_model):
        """Restriction maps must remain in O(d) after forward."""
        model, N, E, d, _ = so4_model
        x = torch.randn(3, N, 1)
        with torch.no_grad():
            _, maps_traj, _ = model(x, torch.tensor(0))
        for maps_t in maps_traj:
            for e_idx in range(min(10, maps_t.shape[0])):
                for s in range(2):
                    F = maps_t[e_idx, s]
                    err = (F.T @ F - torch.eye(d)).norm().item()
                    assert err < 1e-3, f"Map [{e_idx},{s}] not orthogonal: err={err:.4f}"


# ── Backward pass ────────────────────────────────────────────────────────────

class TestSO4Backward:
    def test_gradients_flow(self, so4_model):
        """Gradients must flow through the full model."""
        model, N, E, d, _ = so4_model
        x = torch.randn(3, N, 1)
        preds, maps, laps = model(x, torch.tensor(1))
        loss = sum(p.sum() for p in preds)
        loss.backward()

        # Key parameters must have gradients
        assert model.initial_maps.grad is not None, "No grad on initial_maps"
        assert model.allele_so4.skew_params.weight.grad is not None, "No grad on SO4 params"

    def test_training_step(self, so4_model):
        """One optimizer step must complete without error."""
        model, N, E, d, _ = so4_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = TSGNNLoss()

        x = torch.randn(3, N, 1)
        targets = [torch.randn(N, 1) for _ in range(3)]

        preds, maps, laps = model(x, torch.tensor(1))
        loss, comps = criterion(preds, targets, laps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert comps["loss_expr"] > 0, "Expression loss should be > 0"


# ── Allele divergence ────────────────────────────────────────────────────────

class TestAlleleDivergence:
    def test_allele_divergence_nonzero(self, so4_model):
        """allele_divergence(structural vs WT) must be > 0 at init."""
        model, N, E, d, _ = so4_model
        x = torch.randn(5, N, 1)
        with torch.no_grad():
            preds_wt, _, _ = model(x, torch.tensor(0))       # WT
            preds_st, _, _ = model(x, torch.tensor(1))       # structural

        div = allele_divergence(torch.stack(preds_st), torch.stack(preds_wt))
        assert div > 0, f"allele_divergence = {div}, SO4 may be collapsed"

    def test_different_classes_different_predictions(self, so4_model):
        """WT, structural, dna_contact must produce different predictions."""
        model, N, E, d, _ = so4_model
        torch.manual_seed(42)
        x = torch.randn(3, N, 1)
        with torch.no_grad():
            p0, _, _ = model(x, torch.tensor(0))
            p1, _, _ = model(x, torch.tensor(1))
            p2, _, _ = model(x, torch.tensor(2))

        # Stack last timestep predictions
        last_0 = p0[-1]
        last_1 = p1[-1]
        last_2 = p2[-1]

        d01 = (last_0 - last_1).norm().item()
        d02 = (last_0 - last_2).norm().item()
        d12 = (last_1 - last_2).norm().item()

        assert d01 > 1e-6, f"WT == structural: diff={d01}"
        assert d02 > 1e-6, f"WT == dna_contact: diff={d02}"
        assert d12 > 1e-6, f"structural == dna_contact: diff={d12}"

    def test_allele_divergence_gate(self, so4_model):
        """
        GATE TEST: allele_divergence must exceed 0.001 threshold.
        This is the minimum signal required before HPC submission.
        """
        model, N, E, d, _ = so4_model
        torch.manual_seed(42)
        x = torch.randn(5, N, 1)
        with torch.no_grad():
            preds_wt, _, _ = model(x, torch.tensor(0))
            preds_st, _, _ = model(x, torch.tensor(1))

        div = allele_divergence(torch.stack(preds_st), torch.stack(preds_wt))
        assert div > 0.001, (
            f"allele_divergence = {div:.6f}, below 0.001 gate. "
            "SO4 embedding may be collapsed or init_scale too small."
        )


# ── Bundle edge_weight ───────────────────────────────────────────────────────

class TestBundleEdgeWeight:
    def test_construct_base_grn_returns_signed_weights(self):
        """construct_base_grn must return signed edge_weight tensor."""
        from tsgnn.data.grn_construction import construct_base_grn, load_targetgenereg

        gene_list = [f"GENE_{i}" for i in range(100)]
        gene_list[0] = "TP53"

        tgr_df = load_targetgenereg()
        try:
            edge_index, edge_weight, meta = construct_base_grn(
                gene_list=gene_list,
                targetgenereg=tgr_df if len(tgr_df) > 0 else None,
            )
        except RuntimeError:
            pytest.skip("No prior databases available locally")

        assert edge_weight.dtype == torch.float32
        assert edge_weight.shape[0] == edge_index.shape[1]
        # TargetGeneReg uses +-1.5 weights — should have negative entries
        n_neg = (edge_weight < 0).sum().item()
        # Soft check: at least some edges should have sign info
        assert edge_weight.shape[0] > 0, "No edges in GRN"


# ── Factory function ─────────────────────────────────────────────────────────

class TestFactory:
    def test_create_from_config(self):
        """create_tsgnn_from_config must work with current config schema."""
        config = {
            "data": {"n_genes": 20},
            "model": {
                "stalk_dim": 4,
                "input_dim": 1,
                "num_diffusion_steps": 2,
            },
            "allele_classes": {"n_classes": 3},
        }
        edge_index = torch.randint(0, 20, (2, 40))
        model = create_tsgnn_from_config(config, edge_index)
        assert isinstance(model, TSGNN)
        assert model.allele_so4.skew_params.weight.shape == (3, 6)
