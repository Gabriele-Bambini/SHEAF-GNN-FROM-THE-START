"""Synthetic sheaf-diffusion backtests with known ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from sheaf_ts.core.laplacian_builders import build_connection_laplacian
from sheaf_ts.core.orthogonal import cayley_map
from sheaf_ts.eval.cross_validation import CVSplit
from sheaf_ts.eval.metrics import directional_metrics, fit_mean_drift
from sheaf_ts.hpo.optuna_runner import SearchSpace, run_optuna_cv

__all__ = [
    "SyntheticBacktestResult",
    "SyntheticSheafDataset",
    "backtest_oracle_sheaf",
    "generate_sheaf_diffusion_dataset",
    "synthetic_manifest",
    "tune_tau_with_optuna",
]


@dataclass(frozen=True)
class SyntheticSheafDataset:
    states: torch.Tensor
    deltas: torch.Tensor
    edge_index: torch.Tensor
    maps: torch.Tensor
    edge_weights: torch.Tensor
    laplacian: torch.Tensor
    tau: float
    seed: int


@dataclass(frozen=True)
class SyntheticBacktestResult:
    model_metrics: dict[str, float]
    control_metrics: dict[str, float]
    train_steps: int
    random_seed: int


def _ring_edge_index(num_nodes: int, *, device: torch.device | None = None) -> torch.Tensor:
    edges = []
    for node in range(num_nodes):
        nxt = (node + 1) % num_nodes
        edges.append((node, nxt))
        edges.append((nxt, node))
    if num_nodes > 3:
        for node in range(num_nodes):
            skip = (node + 2) % num_nodes
            edges.append((node, skip))
    return torch.tensor(edges, dtype=torch.long, device=device).T


def _make_orthogonal_maps(
    num_edges: int,
    stalk_dim: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    params = torch.randn(
        num_edges * 2,
        stalk_dim * (stalk_dim - 1) // 2,
        dtype=dtype,
        generator=generator,
    )
    return cayley_map(params, d=stalk_dim).reshape(num_edges, 2, stalk_dim, stalk_dim)


def _predict_delta(states: torch.Tensor, laplacian: torch.Tensor, tau: float) -> torch.Tensor:
    num_steps = states.shape[0] - 1
    num_nodes, stalk_dim = states.shape[1], states.shape[2]
    flat = states[:-1].reshape(num_steps, num_nodes * stalk_dim)
    pred_flat = -float(tau) * (flat @ laplacian.T)
    return pred_flat.reshape(num_steps, num_nodes, stalk_dim)


def generate_sheaf_diffusion_dataset(
    *,
    num_nodes: int = 10,
    stalk_dim: int = 3,
    num_steps: int = 12,
    tau: float = 0.05,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> SyntheticSheafDataset:
    """Generate states from known sheaf diffusion ``x[t+1] = x[t] - tau L x[t]``."""
    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3")
    if stalk_dim < 1:
        raise ValueError("stalk_dim must be positive")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if tau <= 0:
        raise ValueError("tau must be positive")

    generator = torch.Generator().manual_seed(seed)
    edge_index = _ring_edge_index(num_nodes)
    maps = _make_orthogonal_maps(
        edge_index.shape[1],
        stalk_dim,
        generator=generator,
        dtype=dtype,
    )
    edge_weights = 0.2 + 2.8 * torch.rand(
        edge_index.shape[1],
        dtype=dtype,
        generator=generator,
    )
    laplacian = build_connection_laplacian(
        num_nodes=num_nodes,
        edge_index=edge_index,
        maps=maps,
        edge_weights=edge_weights,
        normalize=True,
        eps=0.0,
        verify_orthogonal=True,
    )

    states = [torch.randn(num_nodes, stalk_dim, dtype=dtype, generator=generator)]
    for _ in range(num_steps):
        current = states[-1].reshape(num_nodes * stalk_dim)
        nxt = current - tau * (laplacian @ current)
        states.append(nxt.reshape(num_nodes, stalk_dim))
    stacked = torch.stack(states, dim=0)
    deltas = stacked[1:] - stacked[:-1]
    return SyntheticSheafDataset(
        states=stacked,
        deltas=deltas,
        edge_index=edge_index,
        maps=maps,
        edge_weights=edge_weights,
        laplacian=laplacian,
        tau=float(tau),
        seed=seed,
    )


def _random_graph_laplacian(dataset: SyntheticSheafDataset, *, random_seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(random_seed)
    num_nodes = dataset.states.shape[1]
    num_edges = dataset.edge_index.shape[1]
    random_edge_index = torch.randint(
        low=0,
        high=num_nodes,
        size=(2, num_edges),
        generator=generator,
        dtype=torch.long,
    )
    maps = _make_orthogonal_maps(
        num_edges,
        dataset.states.shape[2],
        generator=generator,
        dtype=dataset.states.dtype,
    )
    edge_weights = 0.2 + 2.8 * torch.rand(
        num_edges,
        dtype=dataset.states.dtype,
        generator=generator,
    )
    return build_connection_laplacian(
        num_nodes=num_nodes,
        edge_index=random_edge_index,
        maps=maps,
        edge_weights=edge_weights,
        normalize=True,
        eps=0.0,
        verify_orthogonal=True,
    )


def backtest_oracle_sheaf(
    dataset: SyntheticSheafDataset,
    *,
    train_steps: int,
    random_seed: int = 0,
) -> SyntheticBacktestResult:
    """Evaluate oracle sheaf predictor against shortcut controls."""
    if train_steps < 1 or train_steps >= dataset.deltas.shape[0]:
        raise ValueError("train_steps must leave at least one test transition")

    train_true = dataset.deltas[:train_steps]
    test_true = dataset.deltas[train_steps:]
    test_state = dataset.states[:-1][train_steps:]
    mean_drift = fit_mean_drift(train_true)

    oracle_pred = _predict_delta(dataset.states, dataset.laplacian, dataset.tau)[train_steps:]
    mean_pred = mean_drift.reshape(1, 1, -1).expand_as(test_true)
    random_laplacian = _random_graph_laplacian(dataset, random_seed=random_seed)
    random_pred = _predict_delta(dataset.states, random_laplacian, dataset.tau)[train_steps:]

    oracle = directional_metrics(
        oracle_pred,
        test_true,
        mean_drift=mean_drift,
        current_state=test_state,
    )
    mean_control = directional_metrics(
        mean_pred,
        test_true,
        mean_drift=mean_drift,
        current_state=test_state,
    )
    random_control = directional_metrics(
        random_pred,
        test_true,
        mean_drift=mean_drift,
        current_state=test_state,
    )

    return SyntheticBacktestResult(
        model_metrics=oracle,
        control_metrics={
            "mean_drift": mean_control["r_delta_perp"],
            "random_graph": random_control["r_delta_perp_no_self"],
        },
        train_steps=train_steps,
        random_seed=random_seed,
    )


def tune_tau_with_optuna(
    dataset: SyntheticSheafDataset,
    *,
    splits: Sequence[CVSplit],
    candidate_taus: Sequence[float],
    n_trials: int,
    seed: int = 0,
) -> dict[str, object]:
    """Tune diffusion ``tau`` on synthetic transitions using the generic Optuna adapter."""
    if not candidate_taus:
        raise ValueError("candidate_taus must be non-empty")

    def objective(params: dict[str, object], split: CVSplit) -> float:
        tau = float(params["tau"])
        pred = _predict_delta(dataset.states, dataset.laplacian, tau)[list(split.val_idx)]
        true = dataset.deltas[list(split.val_idx)]
        mean_drift = fit_mean_drift(dataset.deltas[list(split.train_idx)])
        return directional_metrics(pred, true, mean_drift=mean_drift)["r_delta_perp"]

    return run_optuna_cv(
        objective=objective,
        search_space=SearchSpace(categoricals={"tau": tuple(float(x) for x in candidate_taus)}),
        splits=splits,
        n_trials=n_trials,
        direction="maximize",
        study_name="synthetic_sheaf_tau",
        seed=seed,
    )


def synthetic_manifest(
    dataset: SyntheticSheafDataset,
    result: SyntheticBacktestResult,
    *,
    run_id: str,
) -> dict[str, object]:
    """Build JSON-safe manifest for a synthetic sheaf backtest."""
    model = result.model_metrics["r_delta_perp"]
    mean = result.control_metrics["mean_drift"]
    random = result.control_metrics["random_graph"]
    return {
        "run_id": run_id,
        "dataset": {
            "num_nodes": int(dataset.states.shape[1]),
            "stalk_dim": int(dataset.states.shape[2]),
            "num_steps": int(dataset.deltas.shape[0]),
            "num_edges": int(dataset.edge_index.shape[1]),
            "tau": float(dataset.tau),
            "seed": int(dataset.seed),
        },
        "result": {
            "model_metrics": dict(result.model_metrics),
            "control_metrics": dict(result.control_metrics),
            "train_steps": int(result.train_steps),
            "random_seed": int(result.random_seed),
            "accepted_controls": {
                "mean_drift_margin": float(model - mean),
                "random_graph_margin": float(model - random),
            },
        },
    }
