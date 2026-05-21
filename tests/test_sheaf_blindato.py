import inspect

import pytest
import torch
import torch.nn as nn

from sheaf_ts.core.laplacian_builders import build_connection_laplacian
from sheaf_ts.eval.gates import GateThresholds, evaluate_scientific_gates
from sheaf_ts.experiments.real_data_smoke import _fit_tau_from_features
from sheaf_ts.experiments.real_hpo_smoke import run_real_hpo_smoke
from sheaf_ts.hpc.manifest import _iter_snapshot_files
from tsgnn.model.cayley_orthogonal import CayleyOrthogonalSheafDiffusion
from tsgnn.model.edge_weight_net import WeightedNormalizedOrthogonalLaplacian
from tsgnn.model.temporal_ode import SheafNeuralODE
from tsgnn.training.trainer import TSGNNTrainer


def test_weighted_normalized_orthogonal_laplacian_matches_core_dense_reference():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    maps = torch.eye(2).repeat(2, 2, 1, 1)
    weights = torch.tensor([2.0, 3.0])

    builder = WeightedNormalizedOrthogonalLaplacian(
        num_nodes=3,
        stalk_dim=2,
        edge_index=edge_index,
        degree_regularizer=1e-6,
    )
    observed = builder(maps, weights)
    expected = build_connection_laplacian(
        num_nodes=3,
        edge_index=edge_index,
        maps=maps,
        edge_weights=weights,
        normalize=True,
        degree_regularizer=1e-6,
    )

    assert torch.allclose(observed, expected, atol=1e-5)
    assert torch.linalg.eigvalsh(observed).min() > -1e-5


def test_trainer_passes_temporal_edge_weights_to_model(tmp_path):
    class CapturingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor(1.0))
            self.captured_edge_weights = None

        def forward(
            self,
            node_features_seq,
            allele_embedding,
            edge_weights_seq=None,
            bin_boundaries=None,
            return_stalks=False,
        ):
            self.captured_edge_weights = edge_weights_seq
            pred = node_features_seq[0] * self.param
            lap = torch.eye(1)
            stalk = torch.ones(node_features_seq.shape[1], 1)
            return [pred], [torch.ones(1, 2, 1, 1)], [lap], [stalk]

    class ZeroLoss:
        def __call__(self, **kwargs):
            loss = kwargs["predictions"][0].sum() * 0.0
            return loss, {"mse": 0.0}

    model = CapturingModel()
    trainer = TSGNNTrainer(
        model,
        {"training": {"max_epochs": 1}, "loss": {}},
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
    )
    trainer.criterion = ZeroLoss()

    edge_weights_seq = torch.tensor([[0.25, 0.75]])
    trainer._forward_and_loss(
        node_features_seq=torch.ones(1, 2, 1),
        allele_emb=torch.ones(4),
        targets=[torch.ones(2, 1)],
        viper=None,
        edge_weights_seq=edge_weights_seq,
    )

    assert model.captured_edge_weights is edge_weights_seq


def test_trainer_rejects_nonfinite_validation_loss(tmp_path):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor(1.0))

        def forward(self, node_features_seq, allele_embedding, edge_weights_seq=None, bin_boundaries=None, return_stalks=False):
            pred = node_features_seq[0] * self.param
            return [pred], [torch.ones(1, 2, 1, 1)], [torch.eye(1)], [torch.ones(1, 1)]

    class NanLoss:
        def __call__(self, **kwargs):
            loss = kwargs["predictions"][0].sum() * torch.tensor(float("nan"))
            return loss, {"loss_total": float("nan")}

    trainer = TSGNNTrainer(
        SimpleModel(),
        {"training": {"max_epochs": 1}, "loss": {}},
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
    )
    trainer.criterion = NanLoss()

    with pytest.raises(ValueError, match="Non-finite validation loss"):
        trainer._validate(
            {"WT": {"node_features_seq": torch.ones(1, 1, 1), "targets": [torch.ones(1, 1)]}},
            {"WT": torch.ones(4)},
        )


def test_scientific_gate_rejects_negative_model_even_if_controls_are_worse():
    result = evaluate_scientific_gates(
        model_metric=-0.05,
        controls={"mean_drift": -0.20},
        required=("mean_drift",),
        thresholds=GateThresholds(primary_margin=0.01, minimum_model_metric=0.0),
    )

    assert result["status"] == "REJECT"
    assert "minimum_model_metric" in result["failed_controls"]


def test_scientific_gate_rejects_when_paired_margin_ci_is_too_low():
    result = evaluate_scientific_gates(
        model_metric=0.50,
        controls={"gcn": 0.40},
        required=("gcn",),
        thresholds=GateThresholds(mechanism_margin=0.05, minimum_model_metric=0.0),
        paired_margins={"gcn": [0.10, -0.10, 0.10]},
    )

    assert result["status"] == "REJECT"
    assert result["checks"]["gcn"]["ci_low"] < 0.05


def test_real_hpo_does_not_tune_control_random_seed():
    source = inspect.getsource(run_real_hpo_smoke)
    assert '"random_seed"' not in source


def test_real_smoke_tau_fit_does_not_allow_anti_diffusion():
    features = torch.tensor([1.0, 2.0, 3.0])
    target = -features

    assert _fit_tau_from_features(features, target) == 0.0


def test_runtime_snapshot_includes_production_tsgnn_package():
    root = _iter_snapshot_files.__globals__["Path"](__file__).resolve().parents[1]
    rel_paths = {path.relative_to(root).as_posix() for path in _iter_snapshot_files(root)}
    assert any(path.startswith("src/tsgnn/") for path in rel_paths)


def test_cayley_layer_rejects_raw_matrix_maps_in_v3_path():
    layer = CayleyOrthogonalSheafDiffusion(
        num_nodes=2,
        num_edges=1,
        stalk_dim=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="raw matrix"):
        layer.get_orthogonal_maps(torch.eye(2).reshape(1, 1, 2, 2).repeat(1, 2, 1, 1))


def test_zero_velocity_ode_preserves_state_at_zero_dt():
    ode = SheafNeuralODE(stalk_dim=2, hidden_dim=8)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    maps = torch.randn(1, 2, 2, 2)
    node_features = torch.randn(2, 2)

    observed = ode(maps, node_features, edge_index, delta_t=torch.tensor(0.0))

    assert torch.allclose(observed, maps, atol=1e-6)
