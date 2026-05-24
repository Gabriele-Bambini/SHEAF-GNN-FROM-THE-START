"""
Training Pipeline for TS-GNN.

Phase 3B: Implements the full training loop with:
- Adam optimizer, lr=1e-3
- Early stopping with patience=20 on validation loss
- Gradient clipping (max_norm=1.0)
- Per-allele training (R175H, R273H, R248W, WT)
- Mixed precision (fp16) where supported
- wandb experiment tracking
- Model checkpointing
"""

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: tqdm-like wrapper that does nothing
    def tqdm(iterable, **kwargs):
        return iterable

import torch
import torch.nn as nn

from .loss import TSGNNLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to terminate training when validation loss plateaus."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement. Best val loss: {self.best_loss:.6f}"
                )

    def load_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class TSGNNTrainer:
    """
    Training pipeline for TS-GNN.

    Args:
        model: TS-GNN model instance.
        config: Training configuration dict.
        device: torch device.
        checkpoint_dir: Directory for model checkpoints.
        use_wandb: Enable wandb logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device = None,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        # Move model to device
        self.model.to(self.device)

        # Extract training config
        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("lr", 1e-3)
        self.max_epochs = train_cfg.get("max_epochs", 500)
        self.patience = train_cfg.get("patience", 20)
        # Support both key names for backwards compatibility
        self.max_grad_norm = train_cfg.get("grad_clip_norm", train_cfg.get("max_grad_norm", 1.0))
        # Default to FALSE: fp16 autocast produces NaN in the sheaf Laplacian forward
        # pass (large matmul + eigh + torch.autograd.grad in sparsity loss).
        # Validation loss stays finite because it runs under torch.no_grad() in fp32.
        # Setting mixed_precision=True is only safe on a model known to be fp16-stable.
        self.use_mixed_precision = bool(train_cfg.get("mixed_precision", False))
        if self.use_mixed_precision:
            logger.warning(
                "mixed_precision=True requested. Sheaf diffusion is known to produce "
                "NaN under fp16 autocast on large Laplacians. Proceed with caution."
            )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        # Loss function — all five lambdas wired from config (lambda_4 was previously missing)
        loss_cfg = config.get("loss", {})
        self.criterion = TSGNNLoss(
            lambda_1=loss_cfg.get("lambda_1", 0.1),
            lambda_2=loss_cfg.get("lambda_2", 0.5),
            lambda_3=loss_cfg.get("lambda_3", 0.01),
            lambda_4=loss_cfg.get("lambda_4", 0.0),
            tau=loss_cfg.get("tau", 0.5),
            neuralsort_temperature=loss_cfg.get("neuralsort_temperature", 1.0),
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=self.patience)

        # Mixed precision scaler — torch.amp available from 2.0+; fallback to cuda.amp
        if self.use_mixed_precision and self.device.type == "cuda":
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except TypeError:
                # PyTorch < 2.0 does not accept device string argument
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # wandb
        self.wandb_run = None
        if use_wandb:
            self._init_wandb(config)

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    def _init_wandb(self, config: dict):
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=config.get("wandb_project", "tsgnn"),
                config=config,
                name=config.get("run_name", None),
            )
            wandb.watch(self.model, log="gradients", log_freq=50)
            logger.info("wandb initialized.")
        except ImportError:
            logger.warning("wandb not installed. Logging disabled.")
            self.use_wandb = False

    def train(
        self,
        train_data: Dict[str, dict],
        val_data: Dict[str, dict],
        esm_embeddings: Dict[str, torch.Tensor],
        viper_activity: Optional[Dict[str, torch.Tensor]] = None,
        regulon_mask: Optional[torch.Tensor] = None,
        velocity_fields: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_data: Dict mapping allele -> {
                'node_features_seq': (K, N, input_dim),
                'targets': list of (N, input_dim) tensors,
                'edge_weights_seq': optional (K, E),
            }
            val_data: Same structure for validation.
            esm_embeddings: Dict mapping allele -> (esm_dim,) tensor.
            viper_activity: Optional dict mapping allele -> (K, N) tensor.

        Returns:
            Dict with training history and best metrics.
        """
        logger.info(
            f"Starting training: {self.max_epochs} max epochs, "
            f"patience={self.patience}, lr={self.lr}, device={self.device}"
        )
        logger.info(f"Alleles: {list(train_data.keys())}")

        # Enforce reproducibility from config seed.
        seed = self.config.get("seed", 42)
        import random as _random
        _random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")

        # Optional CUDA determinism (slower, exact bit-for-bit reproducibility).
        # Enable with `training.deterministic: true`. Requires CUBLAS_WORKSPACE_CONFIG
        # env var on CUDA ≥ 10.2 — we set it here so users do not need to remember.
        if self.config.get("training", {}).get("deterministic", False):
            import os as _os
            _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("CUDA deterministic mode ENABLED (slower; bit-for-bit reproducible).")

        best_val_loss = float("inf")
        start_epoch = 0
        start_time = time.time()

        # ── Resume from checkpoint if available ──────────────────────────────
        # Opt-in via training.resume_from_checkpoint (default False) so that a
        # stale or poisoned checkpoint cannot silently resurrect a bad run.
        best_ckpt_path = self.checkpoint_dir / "best.pt"
        resume = self.config.get("training", {}).get("resume_from_checkpoint", False)
        if resume and best_ckpt_path.exists():
            try:
                ckpt = torch.load(str(best_ckpt_path), map_location=self.device,
                                  weights_only=False)
                
                # Check for corrupted checkpoints (NaN/Inf in weights or optimizer states)
                corrupted = False
                for k, v in ckpt["model_state_dict"].items():
                    if torch.is_tensor(v) and torch.is_floating_point(v):
                        if not torch.isfinite(v).all():
                            corrupted = True; break
                
                if corrupted:
                    logger.warning("Checkpoint contains NaN/Inf values! Discarding to prevent pipeline poisoning.")
                    start_epoch = 0
                else:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in ckpt:
                        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    best_val_loss = ckpt.get("val_loss", float("inf"))
                    start_epoch   = ckpt.get("epoch", 0) + 1
                    if "history" in ckpt:
                        self.history = ckpt["history"]
                    logger.info(
                        f"Resumed from checkpoint: epoch={start_epoch - 1}, "
                        f"val_loss={best_val_loss:.6f}"
                    )
            except Exception as e:
                logger.warning(f"Could not resume checkpoint ({e}), starting fresh.")
                start_epoch = 0

        # Always show progress bar if IN_COLAB is somehow set or just show it usually
        # To avoid being disabled in Colab, we remove disable=not sys.stdout.isatty() if IN_COLAB
        epoch_iter = tqdm(
            range(start_epoch, self.max_epochs), desc="Training", unit="epoch",
        )
        for epoch in epoch_iter:
            # Curriculum learning: update lambdas (Task 5.1 & 5.2)
            self._update_lambdas(epoch)

            # Training
            train_loss, train_components = self._train_epoch(
                train_data, esm_embeddings, viper_activity, regulon_mask, velocity_fields
            )
 
            # Validation
            val_loss, val_components = self._validate(
                val_data, esm_embeddings, viper_activity, regulon_mask, velocity_fields
            )

            # Update LR scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            # Logging
            epoch_iter.set_postfix(
                train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                lr=f"{current_lr:.1e}",
            )
            if epoch % 10 == 0 or val_loss < best_val_loss:
                elapsed = time.time() - start_time
                grad_str = f" | Grad: {train_components.get('grad_norm', 0.0):.2f}"
                logger.info(
                    f"Epoch {epoch:4d} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | Time: {elapsed:.0f}s{grad_str}"
                )

            # wandb logging
            if self.use_wandb and self.wandb_run:
                import wandb
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": current_lr,
                }
                for k, v in train_components.items():
                    log_dict[f"train/{k}"] = v
                for k, v in val_components.items():
                    log_dict[f"val/{k}"] = v
                wandb.log(log_dict)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                break

        # Restore best model
        self.early_stopping.load_best(self.model)
        total_time = time.time() - start_time
        logger.info(
            f"Training complete. Best val loss: {best_val_loss:.6f}, "
            f"Total time: {total_time:.0f}s"
        )

        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1 - start_epoch,
            "total_epochs": epoch + 1,
            "total_time": total_time,
            "history": self.history,
        }

    def _train_epoch(
        self,
        train_data: Dict[str, dict],
        esm_embeddings: Dict[str, torch.Tensor],
        viper_activity: Optional[Dict[str, torch.Tensor]] = None,
        regulon_mask: Optional[torch.Tensor] = None,
        velocity_fields: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch over all alleles."""
        self.model.train()
        total_loss = 0.0
        agg_components: Dict[str, float] = {}
        n_alleles = 0

        # Only request gradient on allele_emb if sparsity loss is actually active.
        # Otherwise we waste a graph and risk NaN from torch.autograd.grad on a
        # disabled loss term.
        sparsity_active = (
            getattr(self.criterion, "lambda_3", 0.0) > 0.0
        )

        for allele, data in train_data.items():
            node_features_seq = data["node_features_seq"].to(self.device).float()
            targets = [t.to(self.device).float() for t in data["targets"]]
            allele_emb = esm_embeddings[allele].to(self.device).float()
            bin_bounds_np = data.get("bin_boundaries")
            bin_boundaries = torch.tensor(bin_bounds_np, device=self.device) if bin_bounds_np is not None else None

            # Enable gradient for allele embedding only when sparsity loss is on.
            if sparsity_active:
                allele_emb = allele_emb.detach().requires_grad_(True)
            else:
                allele_emb = allele_emb.detach()

            viper = None
            if viper_activity and allele in viper_activity:
                viper = viper_activity[allele].to(self.device)

            vel = None
            if velocity_fields and allele in velocity_fields:
                vel = [v.to(self.device).float() for v in velocity_fields[allele]]

            self.optimizer.zero_grad()

            if self.scaler is not None:
                _autocast = (torch.amp.autocast if hasattr(torch.amp, "autocast")
                             else torch.cuda.amp.autocast)
                with _autocast("cuda"):
                    loss, components = self._forward_and_loss(
                        node_features_seq, allele_emb, targets, viper, regulon_mask, vel, bin_boundaries
                    )
                if not torch.isfinite(loss):
                    logger.warning(
                        f"Non-finite loss for allele {allele} (fp16 autocast): "
                        f"{loss.item()}. Skipping step."
                    )
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Compute gradient norm before clipping
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
                components["grad_norm"] = total_norm
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, components = self._forward_and_loss(
                    node_features_seq, allele_emb, targets, viper, regulon_mask, vel, bin_boundaries
                )
                if not torch.isfinite(loss):
                    logger.warning(
                        f"Non-finite loss for allele {allele}: {loss.item()}. "
                        "Skipping step."
                    )
                    continue
                loss.backward()
                
                # Compute gradient norm before clipping
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
                components["grad_norm"] = total_norm
                
                self.optimizer.step()

            # Sanitise the loss value before adding to the running total.
            # A single NaN batch would otherwise poison the average and confuse
            # the scheduler/early-stopping. We still report it via the warning
            # above, but keep the epoch total computable.
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                loss_val = 0.0
            total_loss += loss_val
            for k, v in components.items():
                if not math.isfinite(v):
                    v = 0.0
                agg_components[k] = agg_components.get(k, 0.0) + v
            n_alleles += 1

        avg_loss = total_loss / max(n_alleles, 1)
        avg_components = {k: v / max(n_alleles, 1) for k, v in agg_components.items()}
        return avg_loss, avg_components

    def _validate(
        self,
        val_data: Dict[str, dict],
        esm_embeddings: Dict[str, torch.Tensor],
        viper_activity: Optional[Dict[str, torch.Tensor]] = None,
        regulon_mask: Optional[torch.Tensor] = None,
        velocity_fields: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Run validation over all alleles."""
        self.model.eval()
        total_loss = 0.0
        agg_components: Dict[str, float] = {}
        n_alleles = 0

        with torch.no_grad():
            for allele, data in val_data.items():
                node_features_seq = data["node_features_seq"].to(self.device).float()
                targets = [t.to(self.device).float() for t in data["targets"]]
                allele_emb = esm_embeddings[allele].to(self.device).float()
                bin_bounds_np = data.get("bin_boundaries")
                bin_boundaries = torch.tensor(bin_bounds_np, device=self.device) if bin_bounds_np is not None else None

                viper = None
                if viper_activity and allele in viper_activity:
                    viper = viper_activity[allele].to(self.device).float()

                vel = None
                if velocity_fields and allele in velocity_fields:
                    vel = [v.to(self.device).float() for v in velocity_fields[allele]]

                loss, components = self._forward_and_loss(
                    node_features_seq, allele_emb, targets, viper, regulon_mask, vel, bin_boundaries
                )

                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    loss_val = 0.0
                total_loss += loss_val
                for k, v in components.items():
                    if not math.isfinite(v):
                        v = 0.0
                    agg_components[k] = agg_components.get(k, 0.0) + v
                n_alleles += 1

        avg_loss = total_loss / max(n_alleles, 1)
        avg_components = {k: v / max(n_alleles, 1) for k, v in agg_components.items()}
        return avg_loss, avg_components

    def _forward_and_loss(
        self,
        node_features_seq: torch.Tensor,
        allele_emb: torch.Tensor,
        targets: List[torch.Tensor],
        viper: Optional[torch.Tensor],
        regulon_mask: Optional[torch.Tensor] = None,
        velocity_fields: Optional[List[torch.Tensor]] = None,
        bin_boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass + composite loss computation."""
        # Signature: predictions, maps_traj, laplacians, stalks_traj
        out = self.model(
            node_features_seq,
            allele_emb,
            bin_boundaries=bin_boundaries,
            return_stalks=True,
        )
        predictions, maps_traj, laplacians, stalks_traj = out
 
        if not getattr(self, '_checked_psd', False):
            if hasattr(self.model, 'sheaf_layer') and hasattr(self.model.sheaf_layer, 'verify_laplacian_properties'):
                with torch.no_grad():
                    props = self.model.sheaf_layer.verify_laplacian_properties(maps_traj[0])
                logger.info(f"[Init] Laplacian properties: {props}")
                if props.get("min_eigenvalue", 0) < -1e-4:
                    logger.warning(
                        f"[Init] Laplacian NOT PSD: min eigenvalue = {props['min_eigenvalue']:.6f}. "
                        "Consider adding regularisation to the restriction maps."
                    )
            self._checked_psd = True
 
        # Grounded Activity: use Stalk Dimension 0 (TF Activity) as pred_activity
        # stalks_traj is list of K tensors each (N, d)
        # We stack them into (K, N, d) then slice out dim 0 -> (K, N)
        stalks_stack = torch.stack(stalks_traj)  # (K, N, d)
        pred_activity = stalks_stack[:, :, 0]    # (K, N)
 
        return self.criterion(
            predictions=predictions,
            targets=targets,
            sheaf_laplacians=laplacians,
            predicted_activity=pred_activity,
            viper_activity=viper,
            regulon_mask=regulon_mask,
            allele_embedding=allele_emb,
            velocity_fields=velocity_fields,
        )

    def _update_lambdas(self, epoch: int):
        """Update loss weights based on curriculum learning schedule (Task 5.1 & 5.2)."""
        loss_cfg = self.config.get("loss", {})
        lam1_max = loss_cfg.get("lambda_1", 0.1)
        lam2_max = loss_cfg.get("lambda_2", 0.5)
        lam3_max = loss_cfg.get("lambda_3", 0.01)
        lam4_max = loss_cfg.get("lambda_4", 0.0)

        use_curriculum = self.config.get("training", {}).get("curriculum", False)

        if not use_curriculum:
            self.criterion.lambda_1 = lam1_max
            self.criterion.lambda_2 = lam2_max
            self.criterion.lambda_3 = lam3_max
            self.criterion.lambda_4 = lam4_max
            return

        if epoch < 50:
            lam1, lam2, lam3 = 0.0, 0.0, 0.0
        elif epoch < 100:
            scale = (epoch - 50) / 50.0  # 0 -> 1
            lam1, lam2, lam3 = lam1_max * scale, 0.0, 0.0
        elif epoch < 150:
            scale = (epoch - 100) / 50.0
            lam1 = lam1_max
            lam2 = lam2_max * scale
            lam3 = lam3_max * scale * 0.1
        else:
            lam1 = lam1_max
            lam2 = lam2_max
            # Task 5.2 — Lambda annealing per L_sparse
            lam3 = lam3_max * (1 - math.cos(math.pi * epoch / self.max_epochs)) / 2

        self.criterion.lambda_1 = lam1
        self.criterion.lambda_2 = lam2
        self.criterion.lambda_3 = lam3

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save full resumable checkpoint (model + optimizer + scheduler + history)."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "config": self.config,
        }
        path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint (val_loss={val_loss:.6f})")

    def plot_loss_curve(
        self,
        output_dir: str = "figures",
        show: bool = True,
        filename: str = "loss_curve.png",
    ) -> Optional[str]:
        """
        Plot and save the train/val loss curve from self.history.

        Also marks the epoch with the best validation loss and any LR drops.

        Args:
            output_dir: Directory to save the figure.
            show: If True, call plt.show() (renders inline in Colab / Jupyter).
            filename: Output filename.

        Returns:
            Path to saved figure, or None if matplotlib is unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend (safe in Colab/server)
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping loss curve plot.")
            return None

        train_losses = self.history.get("train_loss", [])
        val_losses   = self.history.get("val_loss",   [])
        lrs          = self.history.get("lr",          [])

        if not train_losses:
            logger.warning("No training history to plot.")
            return None

        epochs = list(range(1, len(train_losses) + 1))

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # ── Loss curves ──────────────────────────────────────────────────
        ax1.plot(epochs, train_losses, label="Train loss", color="#2196F3", linewidth=1.8)
        if val_losses:
            ax1.plot(epochs, val_losses, label="Val loss", color="#FF5722", linewidth=1.8)

            # Mark best val loss
            best_val  = min(val_losses)
            best_ep   = val_losses.index(best_val) + 1
            ax1.axvline(x=best_ep, color="#FF5722", linestyle="--", alpha=0.5, linewidth=1)
            ax1.scatter([best_ep], [best_val], color="#FF5722", s=80, zorder=5,
                        label=f"Best val: {best_val:.5f} @ ep {best_ep}")

        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_yscale("log")
        ax1.grid(True, which="both", alpha=0.3)

        # ── LR on secondary axis ─────────────────────────────────────────
        if lrs and len(set(lrs)) > 1:
            ax2 = ax1.twinx()
            ax2.plot(epochs, lrs, color="#9C27B0", linewidth=1, linestyle=":",
                     alpha=0.7, label="LR")
            ax2.set_ylabel("Learning rate", fontsize=10, color="#9C27B0")
            ax2.tick_params(axis="y", labelcolor="#9C27B0")
            ax2.set_yscale("log")
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
        else:
            ax1.legend(fontsize=9, loc="upper right")

        ax1.set_title("TS-GNN Training Loss", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # ── Save ─────────────────────────────────────────────────────────
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = str(out_path / filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Loss curve saved to {save_path}")

        # ── Display inline (Colab / Jupyter) ─────────────────────────────
        if show:
            try:
                # Re-render with interactive backend if in Colab
                from IPython.display import display
                display(fig)
            except ImportError:
                pass  # not in a notebook, skip display

        plt.close(fig)
        return save_path

    def load_checkpoint(self, path: str) -> dict:
        """Load model + full training state from checkpoint. Returns the checkpoint dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        logger.info(
            f"Resumed from checkpoint '{path}' — "
            f"epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f}"
        )
        return checkpoint

    def resume_from_best(self) -> int:
        """
        Load the best checkpoint if it exists. Returns the epoch to resume from (0 if not found).
        Call before trainer.train() to enable automatic resume in Colab.

        Example::

            start_epoch = trainer.resume_from_best()
            # trainer.train() will start from scratch; resumption must be wired in train loop.
        """
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            ckpt = self.load_checkpoint(str(best_path))
            return ckpt["epoch"] + 1
        logger.info("No checkpoint found — starting training from scratch.")
        return 0


def create_synthetic_training_data(
    N: int = 50,
    E: int = 200,
    K: int = 10,
    input_dim: int = 50,
    esm_dim: int = 1280,
    alleles: Optional[List[str]] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Create synthetic training data for development and testing.

    Returns:
        train_data, val_data, esm_embeddings
    """
    if alleles is None:
        alleles = ["WT", "R175H", "R273H", "R248W"]

    train_data = {}
    val_data = {}
    esm_embeddings = {}

    for allele in alleles:
        # Temporal node features
        node_features = torch.randn(K, N, input_dim)
        # Targets: shifted version of features (next-step prediction)
        targets = [node_features[min(t + 1, K - 1)] + 0.1 * torch.randn(N, input_dim)
                    for t in range(K)]

        train_data[allele] = {
            "node_features_seq": node_features,
            "targets": targets,
        }

        # Validation: slightly different data
        val_features = torch.randn(K, N, input_dim)
        val_targets = [val_features[min(t + 1, K - 1)] + 0.1 * torch.randn(N, input_dim)
                       for t in range(K)]
        val_data[allele] = {
            "node_features_seq": val_features,
            "targets": val_targets,
        }

        esm_embeddings[allele] = torch.randn(esm_dim)

    return train_data, val_data, esm_embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..model.tsgnn import TSGNN

    N, E, d, input_dim, K = 50, 200, 4, 50, 10
    esm_dim = 1280

    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N,
        num_edges=E,
        stalk_dim=d,
        input_dim=input_dim,
        esm_dim=esm_dim,
        conditioning_dim=128,
        edge_index=edge_index,
        num_diffusion_steps=2,
    )

    config = {
        "training": {
            "lr": 1e-3,
            "max_epochs": 5,
            "patience": 3,
            "max_grad_norm": 1.0,
            "mixed_precision": False,
        },
        "loss": {
            "lambda_1": 0.1,
            "lambda_2": 0.0,  # No VIPER in synthetic test
            "lambda_3": 0.0,  # No sparsity in synthetic test
            "tau": 0.5,
        },
    }

    train_data, val_data, esm_embeddings = create_synthetic_training_data(
        N=N, E=E, K=K, input_dim=input_dim
    )

    trainer = TSGNNTrainer(model=model, config=config, use_wandb=False)
    results = trainer.train(train_data, val_data, esm_embeddings)

    print(f"\nTraining results:")
    print(f"  Epochs: {results['epochs_trained']}")
    print(f"  Best val loss: {results['best_val_loss']:.6f}")
    print(f"  Time: {results['total_time']:.1f}s")
