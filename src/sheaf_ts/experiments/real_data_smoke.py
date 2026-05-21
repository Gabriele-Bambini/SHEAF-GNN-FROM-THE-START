"""Shortcut-resistant smoke evaluation on cached real TS-GNN tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from sheaf_ts.core.laplacian_builders import build_connection_laplacian
from sheaf_ts.eval.gates import GateThresholds, evaluate_scientific_gates
from sheaf_ts.eval.metrics import directional_metrics, fit_mean_drift, pearsonr_flat

__all__ = [
    "evaluate_real_transition_split",
    "load_real_transition_cache",
    "run_real_data_smoke",
]


def _load_torch(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _first_available_allele(cache: dict[str, Any], requested: str | None) -> str:
    train_data = cache["train_data"]
    if requested and requested in train_data:
        return requested
    if "WT" in train_data:
        return "WT"
    return next(iter(train_data))


def _as_sequence(cache: dict[str, Any], allele: str) -> torch.Tensor:
    entry = cache["train_data"][allele]
    if "node_features_seq" not in entry:
        raise KeyError(f"train_data[{allele!r}] lacks node_features_seq")
    x = entry["node_features_seq"].detach().to(dtype=torch.float64)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError("node_features_seq must have shape (K, N) or (K, N, 1)")
    if x.shape[0] < 3:
        raise ValueError("at least three temporal bins are required")
    if not torch.isfinite(x).all():
        raise ValueError("node_features_seq contains non-finite values")
    return x


def _coerce_edges(grn: dict[str, Any], num_nodes: int, edge_weight_power: float) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = grn["edge_index"].detach().to(dtype=torch.long)
    edge_weight = grn.get("edge_weight")
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float64)
    else:
        edge_weight = edge_weight.detach().to(dtype=torch.float64).reshape(-1).abs()
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains node ids outside the node range")
    if edge_weight.numel() != edge_index.shape[1]:
        raise ValueError("edge_weight length must match edge count")
    if edge_weight_power != 1.0:
        edge_weight = edge_weight.clamp_min(1e-12).pow(edge_weight_power)
    return edge_index, edge_weight


def _coerce_edge_weight_seq(
    entry: dict[str, Any],
    *,
    num_bins: int,
    num_edges: int,
    edge_weight_power: float,
) -> torch.Tensor | None:
    raw = entry.get("edge_weights_seq")
    if raw is None:
        return None
    weights = raw.detach().to(dtype=torch.float64).abs()
    if weights.ndim != 2:
        raise ValueError("edge_weights_seq must have shape (K, E)")
    if weights.shape[0] < num_bins or weights.shape[1] != num_edges:
        raise ValueError("edge_weights_seq shape is incompatible with node_features_seq and edge_index")
    weights = weights[:num_bins]
    if edge_weight_power != 1.0:
        weights = weights.clamp_min(1e-12).pow(edge_weight_power)
    return weights


def load_real_transition_cache(
    *,
    temporal_path: str | Path,
    grn_path: str | Path,
    allele: str | None = "WT",
    edge_weight_power: float = 1.0,
) -> dict[str, Any]:
    temporal = _load_torch(temporal_path)
    grn = _load_torch(grn_path)
    selected = _first_available_allele(temporal, allele)
    entry = temporal["train_data"][selected]
    x_seq = _as_sequence(temporal, selected)
    edge_index, edge_weight = _coerce_edges(grn, x_seq.shape[1], edge_weight_power)
    edge_weight_seq = _coerce_edge_weight_seq(
        entry,
        num_bins=x_seq.shape[0],
        num_edges=edge_index.shape[1],
        edge_weight_power=edge_weight_power,
    )
    return {
        "allele": selected,
        "x_seq": x_seq,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_weight_seq": edge_weight_seq,
        "num_nodes": int(x_seq.shape[1]),
        "num_edges": int(edge_index.shape[1]),
        "num_bins": int(x_seq.shape[0]),
    }


def _signed_maps(edge_index: torch.Tensor, train_delta: torch.Tensor) -> torch.Tensor:
    src, tgt = edge_index
    maps = torch.ones(edge_index.shape[1], 2, 1, 1, dtype=torch.float64)
    if train_delta.shape[0] < 2:
        return maps
    centered = train_delta - train_delta.mean(dim=0, keepdim=True)
    src_delta = centered[:, src]
    tgt_delta = centered[:, tgt]
    cov = (src_delta * tgt_delta).mean(dim=0)
    sign = torch.where(cov < 0, -torch.ones_like(cov), torch.ones_like(cov))
    maps[:, 1, 0, 0] = sign
    return maps


def _identity_maps(num_edges: int) -> torch.Tensor:
    return torch.ones(num_edges, 2, 1, 1, dtype=torch.float64)


def _randomized_edge_index(edge_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, edge_index.shape, generator=generator, dtype=torch.long)


def _fit_tau(x_train: torch.Tensor, y_train: torch.Tensor, laplacian: torch.Tensor) -> float:
    features = torch.stack([-(laplacian @ row) for row in x_train], dim=0)
    denom = float((features * features).sum().item())
    if denom <= 1e-12:
        return 0.0
    tau = float((features * y_train).sum().item() / denom)
    return max(min(tau, 10.0), -10.0)


def _transition_edge_weights(cache: dict[str, Any], index: int, *, use_temporal_edge_weights: bool) -> torch.Tensor:
    edge_weight_seq = cache.get("edge_weight_seq")
    if use_temporal_edge_weights and edge_weight_seq is not None:
        return edge_weight_seq[int(index)]
    return cache["edge_weight"]


def _laplacian_for_transition(
    cache: dict[str, Any],
    *,
    edge_index: torch.Tensor,
    maps: torch.Tensor,
    transition_idx: int,
    degree_regularizer: float,
    use_temporal_edge_weights: bool,
) -> torch.Tensor:
    return build_connection_laplacian(
        num_nodes=cache["num_nodes"],
        edge_index=edge_index,
        maps=maps,
        edge_weights=_transition_edge_weights(
            cache,
            transition_idx,
            use_temporal_edge_weights=use_temporal_edge_weights,
        ),
        normalize=True,
        degree_regularizer=degree_regularizer,
    )


def _diffusion_features(
    x_current: torch.Tensor,
    indices: Sequence[int],
    *,
    cache: dict[str, Any],
    edge_index: torch.Tensor,
    maps: torch.Tensor,
    degree_regularizer: float,
    use_temporal_edge_weights: bool,
) -> torch.Tensor:
    features = []
    for row, transition_idx in zip(x_current, indices, strict=True):
        laplacian = _laplacian_for_transition(
            cache,
            edge_index=edge_index,
            maps=maps,
            transition_idx=int(transition_idx),
            degree_regularizer=degree_regularizer,
            use_temporal_edge_weights=use_temporal_edge_weights,
        )
        features.append(-(laplacian @ row))
    return torch.stack(features, dim=0)


def _fit_tau_dynamic(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_idx: Sequence[int],
    *,
    cache: dict[str, Any],
    edge_index: torch.Tensor,
    maps: torch.Tensor,
    degree_regularizer: float,
    use_temporal_edge_weights: bool,
) -> float:
    features = _diffusion_features(
        x_train,
        train_idx,
        cache=cache,
        edge_index=edge_index,
        maps=maps,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    return _fit_tau_from_features(features, y_train)


def _fit_tau_from_features(features: torch.Tensor, target: torch.Tensor) -> float:
    denom = float((features * features).sum().item())
    if denom <= 1e-12:
        return 0.0
    tau = float((features * target).sum().item() / denom)
    return max(min(tau, 10.0), 0.0)


def _predict_diffusion_dynamic(
    x_current: torch.Tensor,
    indices: Sequence[int],
    *,
    cache: dict[str, Any],
    edge_index: torch.Tensor,
    maps: torch.Tensor,
    tau: float,
    degree_regularizer: float,
    use_temporal_edge_weights: bool,
) -> torch.Tensor:
    features = _diffusion_features(
        x_current,
        indices,
        cache=cache,
        edge_index=edge_index,
        maps=maps,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    return tau * features


def _predict_diffusion(
    x_current: torch.Tensor,
    *,
    laplacian: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    return torch.stack([tau * (-(laplacian @ row)) for row in x_current], dim=0)


def _fit_nodewise_linear(x_train: torch.Tensor, y_train: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_train
    y = y_train
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = (x_centered * x_centered).sum(dim=0) + ridge
    slope = (x_centered * y_centered).sum(dim=0) / denom.clamp_min(1e-12)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _predict_nodewise_linear(x_current: torch.Tensor, slope: torch.Tensor, intercept: torch.Tensor) -> torch.Tensor:
    return x_current * slope + intercept


def _project_out_mean_drift(values: torch.Tensor, mean_drift: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    drift = mean_drift.reshape(-1)
    norm_sq = drift @ drift
    if norm_sq <= eps:
        return values
    rows = values.reshape(-1, values.shape[-1])
    projected = rows - ((rows @ drift) / norm_sq).unsqueeze(1) * drift.unsqueeze(0)
    return projected.reshape_as(values)


def _add_train_fitted_no_self_metrics(
    base_metrics: dict[str, float],
    *,
    pred: torch.Tensor,
    true: torch.Tensor,
    train_current: torch.Tensor,
    train_true: torch.Tensor,
    train_pred: torch.Tensor,
    val_current: torch.Tensor,
    mean_drift: torch.Tensor,
    ridge: float,
) -> dict[str, float]:
    """Remove train-fitted self/autocorrelation from prediction and target.

    This avoids fitting the residualizer on the validation rows themselves,
    which is degenerate when the validation horizon is one or two transitions.
    """
    pred_slope, pred_intercept = _fit_nodewise_linear(train_current, train_pred, ridge)
    true_slope, true_intercept = _fit_nodewise_linear(train_current, train_true, ridge)
    pred_resid = pred - _predict_nodewise_linear(val_current, pred_slope, pred_intercept)
    true_resid = true - _predict_nodewise_linear(val_current, true_slope, true_intercept)
    pred_perp_resid = _project_out_mean_drift(pred_resid, mean_drift)
    true_perp_resid = _project_out_mean_drift(true_resid, mean_drift)
    enriched = dict(base_metrics)
    enriched["r_delta_no_self"] = pearsonr_flat(pred_resid, true_resid)
    enriched["r_delta_perp_no_self"] = pearsonr_flat(pred_perp_resid, true_perp_resid)
    return enriched


def _slice_transitions(x_seq: torch.Tensor, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.tensor(tuple(indices), dtype=torch.long)
    current = x_seq[idx]
    delta = x_seq[idx + 1] - x_seq[idx]
    return current, delta, x_seq[idx + 1]


def evaluate_real_transition_split(
    cache: dict[str, Any],
    *,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    random_seed: int = 123,
    ridge: float = 1e-4,
    degree_regularizer: float = 1e-3,
    use_temporal_edge_weights: bool = True,
) -> dict[str, Any]:
    x_seq = cache["x_seq"]
    num_transitions = x_seq.shape[0] - 1
    if not train_idx or not val_idx:
        raise ValueError("train_idx and val_idx must be non-empty")
    if max(train_idx) >= num_transitions or max(val_idx) >= num_transitions:
        raise ValueError("transition index out of range")

    train_current, train_delta, _ = _slice_transitions(x_seq, train_idx)
    val_current, val_delta, _ = _slice_transitions(x_seq, val_idx)
    mean_drift = fit_mean_drift(train_delta)

    edge_index = cache["edge_index"]
    sheaf_maps = _signed_maps(edge_index, train_delta)
    identity_maps = _identity_maps(edge_index.shape[1])
    sheaf_tau = _fit_tau_dynamic(
        train_current,
        train_delta,
        train_idx,
        cache=cache,
        edge_index=edge_index,
        maps=sheaf_maps,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    sheaf_pred = _predict_diffusion_dynamic(
        val_current,
        val_idx,
        cache=cache,
        edge_index=edge_index,
        maps=sheaf_maps,
        tau=sheaf_tau,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    sheaf_train_pred = _predict_diffusion_dynamic(
        train_current,
        train_idx,
        cache=cache,
        edge_index=edge_index,
        maps=sheaf_maps,
        tau=sheaf_tau,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )

    gcn_tau = _fit_tau_dynamic(
        train_current,
        train_delta,
        train_idx,
        cache=cache,
        edge_index=edge_index,
        maps=identity_maps,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    gcn_pred = _predict_diffusion_dynamic(
        val_current,
        val_idx,
        cache=cache,
        edge_index=edge_index,
        maps=identity_maps,
        tau=gcn_tau,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    gcn_train_pred = _predict_diffusion_dynamic(
        train_current,
        train_idx,
        cache=cache,
        edge_index=edge_index,
        maps=identity_maps,
        tau=gcn_tau,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )

    sheaf_train_features = _diffusion_features(
        train_current,
        train_idx,
        cache=cache,
        edge_index=edge_index,
        maps=sheaf_maps,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    sheaf_val_features = _diffusion_features(
        val_current,
        val_idx,
        cache=cache,
        edge_index=edge_index,
        maps=sheaf_maps,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    sheaf_residual_tau = _fit_tau_from_features(sheaf_train_features, train_delta - gcn_train_pred)
    sheaf_residual_pred = gcn_pred + sheaf_residual_tau * sheaf_val_features
    sheaf_residual_train_pred = gcn_train_pred + sheaf_residual_tau * sheaf_train_features

    random_edge_index = _randomized_edge_index(edge_index, cache["num_nodes"], random_seed)
    random_maps = _identity_maps(random_edge_index.shape[1])
    random_tau = _fit_tau_dynamic(
        train_current,
        train_delta,
        train_idx,
        cache=cache,
        edge_index=random_edge_index,
        maps=random_maps,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    random_pred = _predict_diffusion_dynamic(
        val_current,
        val_idx,
        cache=cache,
        edge_index=random_edge_index,
        maps=random_maps,
        tau=random_tau,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    random_train_pred = _predict_diffusion_dynamic(
        train_current,
        train_idx,
        cache=cache,
        edge_index=random_edge_index,
        maps=random_maps,
        tau=random_tau,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )

    slope, intercept = _fit_nodewise_linear(train_current, train_delta, ridge)
    mlp_pred = _predict_nodewise_linear(val_current, slope, intercept)
    mlp_train_pred = _predict_nodewise_linear(train_current, slope, intercept)
    mean_pred = mean_drift.view(1, -1).expand_as(val_delta)
    mean_train_pred = mean_drift.view(1, -1).expand_as(train_delta)

    def metrics(pred: torch.Tensor, train_pred: torch.Tensor) -> dict[str, float]:
        base = directional_metrics(
            pred,
            val_delta,
            mean_drift=mean_drift,
        )
        return _add_train_fitted_no_self_metrics(
            base,
            pred=pred,
            true=val_delta,
            train_current=train_current,
            train_true=train_delta,
            train_pred=train_pred,
            val_current=val_current,
            mean_drift=mean_drift,
            ridge=ridge,
        )

    model_metrics = {
        "sheaf": metrics(sheaf_pred, sheaf_train_pred),
        "sheaf_residual": metrics(sheaf_residual_pred, sheaf_residual_train_pred),
    }
    baseline_metrics = {
        "mean_drift": metrics(mean_pred, mean_train_pred),
        "random_graph": metrics(random_pred, random_train_pred),
        "mlp": metrics(mlp_pred, mlp_train_pred),
        "gcn": metrics(gcn_pred, gcn_train_pred),
    }
    return {
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "fit": {
            "sheaf_tau": sheaf_tau,
            "sheaf_residual_tau": sheaf_residual_tau,
            "gcn_tau": gcn_tau,
            "random_tau": random_tau,
        },
        "split": {
            "train_idx": list(map(int, train_idx)),
            "val_idx": list(map(int, val_idx)),
        },
        "uses_temporal_edge_weights": bool(use_temporal_edge_weights and cache.get("edge_weight_seq") is not None),
    }


def _default_split(num_bins: int, train_steps: int | None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    num_transitions = num_bins - 1
    if train_steps is None:
        train_steps = max(2, num_transitions - 2)
    train_steps = max(1, min(int(train_steps), num_transitions - 1))
    train_idx = tuple(range(train_steps))
    val_idx = tuple(range(train_steps, num_transitions))
    return train_idx, val_idx


def _gate_from_result(result: dict[str, Any], *, primary_model: str) -> dict[str, Any]:
    metric_name = "r_delta_perp_no_self"
    model_metric = result["model_metrics"][primary_model][metric_name]
    controls = {
        name: metrics[metric_name]
        for name, metrics in result["baseline_metrics"].items()
    }
    return evaluate_scientific_gates(
        model_metric=model_metric,
        controls=controls,
        required=("mean_drift", "random_graph", "mlp", "gcn"),
        thresholds=GateThresholds(primary_margin=0.05, mechanism_margin=0.02),
    )


def run_real_data_smoke(
    *,
    temporal_path: str | Path,
    grn_path: str | Path,
    out_path: str | Path,
    run_id: str,
    allele: str | None = "WT",
    train_steps: int | None = None,
    random_seed: int = 123,
    edge_weight_power: float = 1.0,
    ridge: float = 1e-4,
    degree_regularizer: float = 1e-3,
    use_temporal_edge_weights: bool = True,
    primary_model: str = "sheaf_residual",
    strict: bool = False,
) -> dict[str, Any]:
    cache = load_real_transition_cache(
        temporal_path=temporal_path,
        grn_path=grn_path,
        allele=allele,
        edge_weight_power=edge_weight_power,
    )
    train_idx, val_idx = _default_split(cache["num_bins"], train_steps)
    result = evaluate_real_transition_split(
        cache,
        train_idx=train_idx,
        val_idx=val_idx,
        random_seed=random_seed,
        ridge=ridge,
        degree_regularizer=degree_regularizer,
        use_temporal_edge_weights=use_temporal_edge_weights,
    )
    if primary_model not in result["model_metrics"]:
        raise ValueError(f"unknown primary_model: {primary_model}")
    verdict = _gate_from_result(result, primary_model=primary_model)
    manifest = {
        "run_id": run_id,
        "dataset": {
            "allele": cache["allele"],
            "num_bins": cache["num_bins"],
            "num_nodes": cache["num_nodes"],
            "num_edges": cache["num_edges"],
            "has_temporal_edge_weights": cache.get("edge_weight_seq") is not None,
            "temporal_path": str(temporal_path),
            "grn_path": str(grn_path),
        },
        "hyperparameters": {
            "edge_weight_power": edge_weight_power,
            "ridge": ridge,
            "degree_regularizer": degree_regularizer,
            "random_seed": random_seed,
            "use_temporal_edge_weights": use_temporal_edge_weights,
        },
        "result": result,
        "gate_verdict": verdict,
        "primary_model": primary_model,
        "strict": bool(strict),
    }
    if strict and not verdict["accepted"]:
        raise ValueError("real-data smoke failed strict gates: " + verdict["status"])
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return json.loads(json.dumps(manifest, sort_keys=True))
