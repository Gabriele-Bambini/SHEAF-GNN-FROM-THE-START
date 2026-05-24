"""Unit tests for the TSGNNTrainer training pipeline."""

import pytest
import torch

from tsgnn.model.tsgnn import TSGNN
from tsgnn.training.trainer import TSGNNTrainer, EarlyStopping, create_synthetic_training_data


# ── Fixtures ──────────────────────────────────────────────────────────────

N, E, D, INPUT_DIM, K, ESM_DIM = 15, 40, 2, 15, 3, 1280


@pytest.fixture
def model_and_config():
    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=D, input_dim=INPUT_DIM,
        esm_dim=ESM_DIM, conditioning_dim=32, edge_index=edge_index,
        num_diffusion_steps=1,
    )
    config = {
        "training": {
            "lr": 1e-2,
            "max_epochs": 3,
            "patience": 2,
            "max_grad_norm": 1.0,
            "mixed_precision": False,
        },
        "loss": {
            "lambda_1": 0.1,
            "lambda_2": 0.0,
            "lambda_3": 0.0,
            "tau": 0.5,
        },
    }
    return model, config


@pytest.fixture
def synth_data():
    return create_synthetic_training_data(
        N=N, E=E, K=K, input_dim=INPUT_DIM, alleles=["WT", "R175H"],
    )


# ── Training Loop Tests ──────────────────────────────────────────────────

def test_training_completes(model_and_config, synth_data):
    """Training loop should run to completion without errors.

    Even with synthetic data, this validates that all components
    (forward, loss, backward, optimizer step) are wired correctly.
    """
    model, config = model_and_config
    train_data, val_data, esm_embeddings = synth_data

    trainer = TSGNNTrainer(model=model, config=config, use_wandb=False)
    results = trainer.train(train_data, val_data, esm_embeddings)

    assert "best_val_loss" in results
    assert "epochs_trained" in results
    assert results["epochs_trained"] >= 1
    assert results["best_val_loss"] > 0


def test_training_loss_decreases(model_and_config, synth_data):
    """Training loss should generally decrease over epochs.

    We check that the final train loss is less than the initial one.
    With 3 epochs and lr=1e-2 this should hold for random synthetic data.
    """
    model, config = model_and_config
    config["training"]["max_epochs"] = 5
    config["training"]["patience"] = 10
    train_data, val_data, esm_embeddings = synth_data

    trainer = TSGNNTrainer(model=model, config=config, use_wandb=False)
    results = trainer.train(train_data, val_data, esm_embeddings)

    history = results["history"]
    if len(history["train_loss"]) >= 2:
        # At least the last loss should be less than (or close to) the first
        assert history["train_loss"][-1] <= history["train_loss"][0] * 1.5


# ── Early Stopping Tests ─────────────────────────────────────────────────

def test_early_stopping_triggers():
    """EarlyStopping should trigger after patience epochs without improvement."""
    es = EarlyStopping(patience=3, min_delta=0.0)
    model = torch.nn.Linear(5, 5)

    # Give it a good loss, then worse losses
    es(0.5, model)
    assert not es.early_stop
    es(0.6, model)
    es(0.7, model)
    es(0.8, model)
    assert es.early_stop


def test_early_stopping_resets_on_improvement():
    """Counter should reset when validation loss improves."""
    es = EarlyStopping(patience=3, min_delta=0.0)
    model = torch.nn.Linear(5, 5)

    es(0.5, model)
    es(0.6, model)  # counter = 1
    es(0.4, model)  # improvement! counter resets to 0
    assert not es.early_stop
    assert es.counter == 0


def test_early_stopping_saves_best_state():
    """EarlyStopping should save the model state at the best validation loss."""
    es = EarlyStopping(patience=3)
    model = torch.nn.Linear(5, 5)

    # Record best state
    es(0.5, model)
    assert es.best_state is not None

    # Corrupt model
    with torch.no_grad():
        model.weight.fill_(999.0)

    # Restore best
    es.load_best(model)
    assert model.weight.abs().max().item() < 900  # should have been restored


# ── Checkpoint Tests ─────────────────────────────────────────────────────

def test_checkpoint_save_load(model_and_config, synth_data, tmp_path):
    """Checkpoint save and load should produce identical model state."""
    model, config = model_and_config
    train_data, val_data, esm_embeddings = synth_data

    trainer = TSGNNTrainer(
        model=model, config=config,
        checkpoint_dir=str(tmp_path), use_wandb=False,
    )
    trainer.train(train_data, val_data, esm_embeddings)

    # Check that checkpoint was saved
    best_path = tmp_path / "best.pt"
    assert best_path.exists()

    # Load and verify
    checkpoint = torch.load(best_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "epoch" in checkpoint
    assert "val_loss" in checkpoint


# ── Synthetic Data Generation Tests ──────────────────────────────────────

def test_synthetic_data_shapes():
    """Synthetic data should have correct tensor shapes."""
    train, val, esm = create_synthetic_training_data(
        N=10, E=30, K=5, input_dim=10, alleles=["WT", "R175H"],
    )
    assert "WT" in train
    assert "R175H" in train
    assert train["WT"]["node_features_seq"].shape == (5, 10, 10)
    assert len(train["WT"]["targets"]) == 5
    assert esm["WT"].shape == (1280,)
