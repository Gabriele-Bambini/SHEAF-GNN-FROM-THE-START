"""
Optuna hyperparameter optimisation for TS-GNN.

Searches over:
  - Learning rate
  - Stalk dimension (d)
  - FiLM conditioning dim
  - Loss weights (λ₁, λ₂, λ₃)
  - Number of diffusion steps

Usage (CLI)::

    python -m tsgnn.training.optuna_search \
        --n-trials 50 \
        --timeout 3600 \
        --data-dir /path/to/data \
        --out-config configs/optuna_best.yaml

Typical Colab usage::

    from tsgnn.training.optuna_search import run_optuna_search
    best = run_optuna_search(train_data, val_data, esm_embeddings, n_trials=30)
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)


# ── Search Space ─────────────────────────────────────────────────────────────

def _suggest_config(trial, base_config: dict) -> dict:
    """Build a config dict from Optuna trial suggestions."""
    cfg = copy.deepcopy(base_config)

    # Learning rate (log scale)
    cfg.setdefault("training", {})["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    # Stalk dimension: categorical {2, 4, 8}
    cfg.setdefault("model", {})["stalk_dim"] = trial.suggest_categorical("stalk_dim", [2, 4, 8])

    # FiLM conditioning dim
    cfg["model"]["conditioning_dim"] = trial.suggest_categorical(
        "conditioning_dim", [64, 128, 256]
    )

    # Sheaf diffusion steps
    cfg["model"]["num_diffusion_steps"] = trial.suggest_int("num_diffusion_steps", 1, 5)

    # Loss weights (log scale)
    cfg.setdefault("loss", {})["lambda_1"] = trial.suggest_float(
        "lambda_1", 0.01, 1.0, log=True
    )
    cfg["loss"]["lambda_2"] = trial.suggest_float("lambda_2", 0.1, 2.0, log=True)
    cfg["loss"]["lambda_3"] = trial.suggest_float("lambda_3", 1e-3, 0.1, log=True)

    # Hinge threshold τ
    cfg["loss"]["tau"] = trial.suggest_float("tau", 0.1, 1.0)

    # Weight decay
    cfg["training"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 0.0, 1e-3
    )

    # Auto-enable gradient checkpointing for large stalk dim
    if cfg["model"]["stalk_dim"] >= 8:
        cfg["model"]["use_gradient_checkpointing"] = True

    return cfg


# ── Objective ────────────────────────────────────────────────────────────────

def _objective(
    trial,
    base_config: dict,
    train_data: dict,
    val_data: dict,
    esm_embeddings: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    n_epochs_per_trial: int,
    device: torch.device,
):
    """Optuna objective: returns best validation loss for a trial."""
    import optuna

    cfg = _suggest_config(trial, base_config)

    # Build model
    from tsgnn.model.tsgnn import create_tsgnn_from_config
    from tsgnn.training.trainer import TSGNNTrainer

    try:
        model = create_tsgnn_from_config(cfg, edge_index)
    except Exception as e:
        logger.warning(f"Trial {trial.number}: model creation failed — {e}")
        raise optuna.exceptions.TrialPruned()

    # Short training run
    cfg_short = copy.deepcopy(cfg)
    cfg_short.setdefault("training", {})["max_epochs"] = n_epochs_per_trial
    cfg_short["training"]["patience"] = max(5, n_epochs_per_trial // 4)

    trainer = TSGNNTrainer(
        model=model,
        config=cfg_short,
        device=device,
        checkpoint_dir=f"/tmp/tsgnn_optuna/trial_{trial.number}",
        use_wandb=False,
    )

    best_val_loss = float("inf")
    for epoch in range(n_epochs_per_trial):
        # _train_epoch / _validate return (avg_loss, components) tuples
        train_loss, _ = trainer._train_epoch(train_data, esm_embeddings, None)
        val_loss, _ = trainer._validate(val_data, esm_embeddings, None)

        # Report to Optuna for pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss


# ── Public API ────────────────────────────────────────────────────────────────

def run_optuna_search(
    train_data: dict,
    val_data: dict,
    esm_embeddings: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    base_config: Optional[dict] = None,
    n_trials: int = 50,
    n_epochs_per_trial: int = 30,
    timeout: Optional[int] = None,
    device: Optional[torch.device] = None,
    out_config_path: Optional[str] = None,
    study_name: str = "tsgnn_optuna",
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search for TS-GNN.

    Args:
        train_data: Dict mapping allele → TemporalGraphSequence (training split).
        val_data:   Dict mapping allele → TemporalGraphSequence (validation split).
        esm_embeddings: Dict mapping allele → (1280,) ESM-2 tensor.
        base_config: Base config dict to override. Loads configs/default.yaml if None.
        n_trials: Number of Optuna trials (default 50).
        n_epochs_per_trial: Training epochs per trial (default 30).
        timeout: Stop search after this many seconds (optional).
        device: Torch device (autodetects if None).
        out_config_path: Save best config to this YAML path.
        study_name: Optuna study name (for persistence).
        storage: Optuna storage URL (e.g. "sqlite:///optuna.db") for resumable search.

    Returns:
        best_params: Dict of best hyperparameters found.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter search. "
            "Install with: pip install optuna"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base config. Try repo layout first, then CWD — both return empty
    # dict if neither is found so the caller still gets a valid config.
    if base_config is None:
        here = Path(__file__).resolve()
        candidates = []
        try:
            candidates.append(here.parents[3] / "configs" / "default.yaml")
        except IndexError:
            pass
        candidates.append(Path.cwd() / "configs" / "default.yaml")
        base_config = {}
        for config_path in candidates:
            if config_path.is_file():
                with open(config_path) as f:
                    base_config = yaml.safe_load(f) or {}
                break

    # Create study with median pruning
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=10, interval_steps=1
    )
    sampler = optuna.samplers.TPESampler(seed=base_config.get("seed", 42))

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        load_if_exists=(storage is not None),
    )

    logger.info(
        f"Starting Optuna search: {n_trials} trials, "
        f"{n_epochs_per_trial} epochs each, device={device}"
    )

    def objective(trial):
        return _objective(
            trial,
            base_config=base_config,
            train_data=train_data,
            val_data=val_data,
            esm_embeddings=esm_embeddings,
            edge_index=edge_index,
            n_epochs_per_trial=n_epochs_per_trial,
            device=device,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
    )

    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Best trial: val_loss={best_value:.6f}")
    logger.info(f"Best params: {best_params}")

    # Build and save the best config
    best_cfg = _suggest_config(study.best_trial, base_config)

    if out_config_path is not None:
        out_path = Path(out_config_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Best config saved to {out_path}")

    return {
        "best_params": best_params,
        "best_value": best_value,
        "best_config": best_cfg,
        "study": study,
    }


def print_study_summary(study) -> None:
    """Print a formatted summary of an Optuna study."""
    try:
        import optuna
    except ImportError:
        return

    print(f"\n{'='*60}")
    print(f"  Optuna Study: {study.study_name}")
    print(f"{'='*60}")
    print(f"  Trials completed : {len(study.trials)}")
    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"  Trials pruned    : {pruned}")
    print(f"  Best val loss    : {study.best_value:.6f}")
    print(f"\n  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k:30s}: {v}")
    print(f"{'='*60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="TS-GNN Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=30,
                        help="Training epochs per trial")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max wall-clock seconds for search")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="TSGNN_DATA_DIR override")
    parser.add_argument("--out-config", type=str, default="configs/optuna_best.yaml",
                        help="Where to save the best config YAML")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    parser.add_argument("--study-name", type=str, default="tsgnn_optuna")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["TSGNN_DATA_DIR"] = args.data_dir

    logger.info("Loading data for hyperparameter search...")
    logger.info("(Provide train_data/val_data/esm_embeddings programmatically "
                "via run_optuna_search() for full pipeline integration)")

    logger.info(
        "\nTo run the full search, use run_optuna_search() directly:\n"
        "  from tsgnn.training.optuna_search import run_optuna_search\n"
        "  results = run_optuna_search(\n"
        "      train_data=train_sequences,\n"
        "      val_data=val_sequences,\n"
        "      esm_embeddings=esm_embeddings,\n"
        f"     n_trials={args.n_trials},\n"
        f"     n_epochs_per_trial={args.n_epochs},\n"
        f"     out_config_path='{args.out_config}',\n"
        "  )\n"
    )
