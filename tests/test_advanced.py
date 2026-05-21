"""
Advanced tests for new components:
- NormalizedVectorizedSheafDiffusion
- NeuralSort differentiable ranking
- Velocity consistency loss
- DoRothEA prior edge loading
- Checkpoint save/resume
- Optuna search (1 trial smoke test)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn.functional as F


# ── Normalized Sheaf Laplacian ────────────────────────────────────────────────

class TestNormalizedSheafLaplacian:
    @pytest.fixture
    def norm_layer(self):
        from tsgnn.model.sheaf_vectorized import NormalizedVectorizedSheafDiffusion
        N, E, d = 8, 16, 3
        torch.manual_seed(0)
        ei = torch.randint(0, N, (2, E))
        return NormalizedVectorizedSheafDiffusion(N, E, d, ei), N, E, d

    def test_eigenvalue_range(self, norm_layer):
        """Eigenvalues of normalized Laplacian must lie in [0, 2]."""
        layer, N, E, d = norm_layer
        maps = torch.randn(E, 2, d, d)
        L = layer.compute_connection_laplacian(maps)
        vals = torch.linalg.eigvalsh(L)
        assert vals.min() >= -1e-3, f"Min eigenvalue {vals.min():.4f} < 0"
        assert vals.max() <= 2.0 + 1e-3, f"Max eigenvalue {vals.max():.4f} > 2"

    def test_symmetry(self, norm_layer):
        """Normalized Laplacian must be symmetric."""
        layer, N, E, d = norm_layer
        maps = torch.randn(E, 2, d, d)
        L = layer.compute_connection_laplacian(maps)
        assert torch.allclose(L, L.T, atol=1e-4), "Normalized Laplacian not symmetric"

    def test_psd(self, norm_layer):
        """Normalized Laplacian must be positive semi-definite."""
        layer, N, E, d = norm_layer
        maps = torch.randn(E, 2, d, d)
        L = layer.compute_connection_laplacian(maps)
        vals = torch.linalg.eigvalsh(L)
        assert vals.min() >= -1e-3, f"Not PSD: min eigenvalue {vals.min():.6f}"

    def test_api_alias(self, norm_layer):
        """compute_connection_laplacian and _vectorized must give same result."""
        layer, N, E, d = norm_layer
        maps = torch.randn(E, 2, d, d)
        L1 = layer.compute_connection_laplacian(maps)
        L2 = layer.compute_connection_laplacian_vectorized(maps)
        assert torch.allclose(L1, L2, atol=1e-6)

    def test_diffuse_shape(self, norm_layer):
        """diffuse() must preserve (N, d) shape."""
        layer, N, E, d = norm_layer
        maps = torch.randn(E, 2, d, d)
        L = layer.compute_connection_laplacian(maps)
        x = torch.randn(N, d)
        out = layer.diffuse(x, L)
        assert out.shape == (N, d)


# ── NeuralSort ────────────────────────────────────────────────────────────────

class TestNeuralSort:
    def test_rank_ordering(self):
        """Larger values must receive smaller (better) rank numbers."""
        from tsgnn.training.loss import _neural_sort_ranks
        x = torch.tensor([1.0, 5.0, 3.0])
        r = _neural_sort_ranks(x, tau=0.5)
        # 5.0 is largest -> rank 1, 3.0 -> rank 2, 1.0 -> rank 3
        assert r[1] < r[2] < r[0], f"Rank ordering wrong: {r.tolist()}"

    def test_rank_range(self):
        """Soft ranks must lie in [1, N]."""
        from tsgnn.training.loss import _neural_sort_ranks
        x = torch.randn(10)
        r = _neural_sort_ranks(x, tau=1.0)
        assert r.min() >= 0.9, f"Rank below 1: {r.min():.4f}"
        assert r.max() <= 10.1, f"Rank above N: {r.max():.4f}"

    def test_differentiable(self):
        """NeuralSort must be differentiable w.r.t. input."""
        from tsgnn.training.loss import _neural_sort_ranks
        x = torch.randn(5, requires_grad=True)
        r = _neural_sort_ranks(x, tau=1.0)
        r.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_spearman_monotone(self):
        """Differentiable Spearman of identical inputs must be ~1."""
        from tsgnn.training.loss import _differentiable_spearman
        x = torch.arange(20).float()
        corr = _differentiable_spearman(x, x, temperature=1.0)
        assert corr.item() > 0.95, f"Spearman(x,x) = {corr.item():.4f}, expected ~1"


# ── Velocity Consistency Loss ─────────────────────────────────────────────────

class TestVelocityLoss:
    def test_zero_when_no_velocity(self):
        """Loss must be 0 when velocity_fields is None."""
        from tsgnn.training.loss import TSGNNLoss
        K, N = 5, 20
        preds = [torch.randn(N, N) for _ in range(K)]
        targets = [torch.randn(N, N) for _ in range(K)]
        L_dummy = [torch.eye(N * 4) for _ in range(K)]
        crit = TSGNNLoss(lambda_4=0.5)
        _, comps = crit(preds, targets, L_dummy, velocity_fields=None)
        assert comps["loss_vel"] == 0.0

    def test_activated_by_lambda4(self):
        """Loss must be nonzero when velocity provided and lambda_4 > 0."""
        from tsgnn.training.loss import TSGNNLoss
        K, N = 4, 15
        preds = [torch.randn(N, N) for _ in range(K)]
        targets = preds
        L_dummy = [torch.eye(N * 4) for _ in range(K)]
        vels = [torch.randn(N, N) for _ in range(K - 1)]
        crit = TSGNNLoss(lambda_4=1.0)
        total, comps = crit(preds, targets, L_dummy, velocity_fields=vels)
        assert comps["loss_vel"] != 0.0

    def test_perfect_alignment_negative(self):
        """Perfectly aligned predictions should give negative (minimized) velocity loss."""
        from tsgnn.training.loss import TSGNNLoss
        N = 10
        # Velocity points in +x direction, predictions move in +x direction -> aligned
        vel = torch.zeros(N, N); vel[:, 0] = 1.0
        p0 = torch.zeros(N, N)
        p1 = torch.zeros(N, N); p1[:, 0] = 1.0  # delta = p1 - p0 = vel -> cos sim = 1
        crit = TSGNNLoss(lambda_4=1.0)
        L_dummy = [torch.eye(N * 4)] * 2
        _, comps = crit([p0, p1], [p0, p1], L_dummy, velocity_fields=[vel])
        # -cos_similarity should be ~-1 (minimised when perfectly aligned)
        assert comps["loss_vel"] < 0.0


# ── DoRothEA Prior Edges ──────────────────────────────────────────────────────

class TestDorotheaPriors:
    def test_returns_dict(self):
        """load_dorothea_prior_edges must return a dict (empty if no decoupler)."""
        from tsgnn.data.grn_construction import load_dorothea_prior_edges
        gene_list = ["TP53", "FOXA1", "GATA3", "ESR1", "MYC", "BRCA1"]
        result = load_dorothea_prior_edges(gene_list, levels=["A", "B"])
        assert isinstance(result, dict)

    def test_no_self_loops(self):
        """DoRothEA priors must not contain self-loops."""
        from tsgnn.data.grn_construction import load_dorothea_prior_edges
        gene_list = [f"GENE_{i}" for i in range(20)]
        result = load_dorothea_prior_edges(gene_list, levels=["A"])
        for (src, tgt) in result:
            assert src != tgt, f"Self-loop at {src}"

    def test_indices_in_range(self):
        """All edge indices must be valid gene_list indices."""
        from tsgnn.data.grn_construction import load_dorothea_prior_edges
        gene_list = ["TP53", "MYC", "BRCA1", "ESR1"]
        result = load_dorothea_prior_edges(gene_list, levels=["A", "B"])
        N = len(gene_list)
        for (src, tgt) in result:
            assert 0 <= src < N
            assert 0 <= tgt < N


# ── Checkpoint Resume ─────────────────────────────────────────────────────────

class TestCheckpointResume:
    def test_save_and_load(self, tmp_path):
        """Saved checkpoint must restore model weights and optimizer state."""
        from tsgnn.model.tsgnn import TSGNN
        from tsgnn.training.trainer import TSGNNTrainer
        import torch

        N, E, d = 10, 20, 2
        ei = torch.randint(0, N, (2, E))
        model = TSGNN(
            num_nodes=N, num_edges=E, stalk_dim=d,
            input_dim=N, esm_dim=32, conditioning_dim=16, edge_index=ei,
        )
        config = {
            "training": {"lr": 1e-3, "max_epochs": 2, "patience": 5,
                         "grad_clip_norm": 1.0},
            "loss": {"lambda_1": 0.1, "lambda_2": 0.5, "lambda_3": 0.01, "tau": 0.5},
        }
        trainer = TSGNNTrainer(model, config, checkpoint_dir=str(tmp_path))

        # Save a checkpoint manually
        trainer._save_checkpoint(epoch=5, val_loss=0.42, is_best=True)

        # Perturb weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        # Restore
        ckpt = trainer.load_checkpoint(str(tmp_path / "best.pt"))
        assert ckpt["epoch"] == 5
        assert abs(ckpt["val_loss"] - 0.42) < 1e-6

        # Verify weights restored (not 999)
        for p in model.parameters():
            assert p.abs().max().item() < 500, "Weights not restored after load_checkpoint"

    def test_resume_from_best_returns_zero_if_no_ckpt(self, tmp_path):
        """resume_from_best must return 0 when no checkpoint exists."""
        from tsgnn.model.tsgnn import TSGNN
        from tsgnn.training.trainer import TSGNNTrainer

        N, E, d = 6, 10, 2
        ei = torch.randint(0, N, (2, E))
        model = TSGNN(N, E, d, N, 32, 16, ei)
        config = {
            "training": {"lr": 1e-3, "max_epochs": 2, "patience": 5,
                         "grad_clip_norm": 1.0},
            "loss": {"lambda_1": 0.1, "lambda_2": 0.5, "lambda_3": 0.01, "tau": 0.5},
        }
        trainer = TSGNNTrainer(model, config, checkpoint_dir=str(tmp_path))
        start = trainer.resume_from_best()
        assert start == 0


# ── Optuna Smoke Test ────────────────────────────────────────────────────────

class TestOptuna:
    def test_suggest_config_has_all_keys(self):
        """_suggest_config must return a dict with all required model keys."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        from tsgnn.training.optuna_search import _suggest_config

        study = optuna.create_study(direction="minimize")

        def objective(trial):
            cfg = _suggest_config(trial, base_config={
                "model": {"stalk_dim": 4, "input_dim": 50, "esm_dim": 1280,
                           "conditioning_dim": 128, "num_diffusion_steps": 3},
                "training": {"lr": 1e-3, "weight_decay": 0.0},
                "loss": {"lambda_1": 0.1, "lambda_2": 0.5, "lambda_3": 0.01, "tau": 0.5},
            })
            assert "model" in cfg
            assert "stalk_dim" in cfg["model"]
            assert "lr" in cfg["training"]
            assert "lambda_1" in cfg["loss"]
            return 0.0

        study.optimize(objective, n_trials=1)
        assert len(study.trials) == 1
