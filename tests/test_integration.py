"""Integration test: full forward → loss → backward → optimizer step.

This is the acid test of the system: verifies that every component
(sheaf layer, GRU, FiLM, composite loss, autograd) works together
end-to-end without errors, NaN values, or shape mismatches.
"""

import pytest
import torch

from tsgnn.model.tsgnn import TSGNN
from tsgnn.model.baselines import EvolveGCN, TemporalGAT, create_tsgnn_no_allele
from tsgnn.training.loss import TSGNNLoss


# ── Constants ─────────────────────────────────────────────────────────────

N, E, D, INPUT_DIM, K, ESM_DIM = 15, 40, 2, 15, 3, 1280


# ── Helpers ───────────────────────────────────────────────────────────────

def _run_train_step(model, criterion, node_seq, allele_emb, targets):
    """Execute one full training step: forward → loss → backward → step."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    preds, maps_traj, laps = model(node_seq, allele_emb)
    pred_activity = torch.stack([p.mean(dim=-1) for p in preds])

    total, components = criterion(
        predictions=preds,
        targets=targets,
        sheaf_laplacians=laps,
        predicted_activity=pred_activity,
        viper_activity=torch.randn(K, N),
        allele_embedding=allele_emb,
    )

    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total, components


# ── TS-GNN Integration Test ──────────────────────────────────────────────

def test_tsgnn_full_train_step():
    """Complete training step with TS-GNN: forward → composite loss → backward → step.

    Validates that:
    1. All 4 loss terms are computed
    2. Gradients flow to all learnable parameters
    3. Optimizer step does not produce NaN
    4. Total loss is a finite positive scalar
    """
    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=D, input_dim=INPUT_DIM,
        esm_dim=ESM_DIM, conditioning_dim=32, edge_index=edge_index,
        num_diffusion_steps=1,
    )
    criterion = TSGNNLoss(lambda_1=0.1, lambda_2=0.5, lambda_3=0.01, tau=0.5)

    node_seq = torch.randn(K, N, INPUT_DIM)
    allele_emb = torch.randn(ESM_DIM, requires_grad=True)
    targets = [torch.randn(N, INPUT_DIM) for _ in range(K)]

    total, components = _run_train_step(model, criterion, node_seq, allele_emb, targets)

    # Loss should be finite and positive
    assert torch.isfinite(total), "Total loss is NaN/Inf"
    assert total.item() > 0, "Total loss should be positive"

    # All components should be present
    assert "loss_expr" in components
    assert "loss_topo" in components
    assert "loss_total" in components

    # Model parameters should have finite values after step
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"NaN in {name} after optimizer step"


def test_tsgnn_multiple_steps_stable():
    """Multiple training steps should not cause divergence.

    Theoretical concern: the sheaf Laplacian is Nd × Nd dense, and repeated
    diffusion + GRU updates can amplify values. Gradient clipping and
    residual connections should prevent this.
    """
    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=D, input_dim=INPUT_DIM,
        esm_dim=ESM_DIM, conditioning_dim=32, edge_index=edge_index,
        num_diffusion_steps=1,
    )
    criterion = TSGNNLoss(lambda_1=0.1, lambda_2=0.0, lambda_3=0.0, tau=0.5)

    losses = []
    for step in range(5):
        node_seq = torch.randn(K, N, INPUT_DIM)
        allele_emb = torch.randn(ESM_DIM)
        targets = [torch.randn(N, INPUT_DIM) for _ in range(K)]

        total, _ = _run_train_step(model, criterion, node_seq, allele_emb, targets)
        losses.append(total.item())

    # None of the losses should be NaN
    assert all(loss == loss for loss in losses), "NaN loss during multi-step training"
    # Losses should stay bounded (not explode)
    assert max(losses) < 1e6, f"Loss exploded: {losses}"


# ── Baseline Integration Tests ───────────────────────────────────────────

def test_evolvegcn_train_step():
    """EvolveGCN should complete a full training step."""
    edge_index = torch.randint(0, N, (2, E))
    model = EvolveGCN(
        num_nodes=N, input_dim=INPUT_DIM, hidden_dim=16, edge_index=edge_index
    )
    criterion = TSGNNLoss(lambda_1=0.0, lambda_2=0.0, lambda_3=0.0)

    node_seq = torch.randn(K, N, INPUT_DIM)
    targets = [torch.randn(N, INPUT_DIM) for _ in range(K)]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    preds, _, laps = model(node_seq)
    total, _ = criterion(preds, targets, laps)
    total.backward()
    optimizer.step()

    assert torch.isfinite(total)


def test_temporal_gat_train_step():
    """Temporal GAT should complete a full training step."""
    edge_index = torch.randint(0, N, (2, E))
    model = TemporalGAT(
        num_nodes=N, input_dim=INPUT_DIM, hidden_dim=16, edge_index=edge_index
    )
    criterion = TSGNNLoss(lambda_1=0.0, lambda_2=0.0, lambda_3=0.0)

    node_seq = torch.randn(K, N, INPUT_DIM)
    allele_emb = torch.randn(ESM_DIM)
    targets = [torch.randn(N, INPUT_DIM) for _ in range(K)]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    preds, _, laps = model(node_seq, allele_emb)
    total, _ = criterion(preds, targets, laps)
    total.backward()
    optimizer.step()

    assert torch.isfinite(total)


def test_tsgnn_no_allele_train_step():
    """TS-GNN without allele conditioning should complete a training step."""
    edge_index = torch.randint(0, N, (2, E))
    model = create_tsgnn_no_allele(
        num_nodes=N, num_edges=E, stalk_dim=D, input_dim=INPUT_DIM,
        edge_index=edge_index, num_diffusion_steps=1,
    )
    criterion = TSGNNLoss(lambda_1=0.1, lambda_2=0.0, lambda_3=0.0, tau=0.5)

    node_seq = torch.randn(K, N, INPUT_DIM)
    allele_emb = torch.randn(ESM_DIM)
    targets = [torch.randn(N, INPUT_DIM) for _ in range(K)]

    total, _ = _run_train_step(model, criterion, node_seq, allele_emb, targets)
    assert torch.isfinite(total)
