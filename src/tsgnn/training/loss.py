"""
Composite 4-Term Loss Function for TS-GNN Training.

Phase 3A: L = L_expr + lambda_1 * L_topo + lambda_2 * L_regulon + lambda_3 * L_sparse
                     + lambda_4 * L_vel

Terms:
1. L_expr:    MSE between predicted and observed gene expression across K time steps
2. L_topo:    Hinge penalty on excessive edge weight changes between consecutive steps
3. L_regulon: Negative Spearman correlation between predicted and VIPER TF activity
4. L_sparse:  L1 norm of dL_F/dz_allele (allele-specific Laplacian perturbation sparsity)
5. L_vel:     Negative mean cosine similarity between predicted trajectory deltas and RNA velocity
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TSGNNLoss(nn.Module):
    """
    Composite loss for TS-GNN training.

    Args:
        lambda_1: Weight for topological preservation loss.
        lambda_2: Weight for regulon activity loss.
        lambda_3: Weight for sparsity loss.
        lambda_4: Weight for velocity consistency loss (default 0.0, inactive unless
                  velocity_fields are supplied).
        tau: Hinge threshold for topological change rate.
    """

    def __init__(
        self,
        lambda_1: float = 0.1,
        lambda_2: float = 0.5,
        lambda_3: float = 0.01,
        lambda_4: float = 0.0,
        tau: float = 0.5,
        neuralsort_temperature: float = 1.0,
    ):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.tau = tau
        self.neuralsort_temperature = neuralsort_temperature

    def expression_loss(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        L_expr = (1/NK) * sum_t ||X_hat(t) - X_obs(t)||^2_F

        Mean squared Frobenius norm across all K time steps.

        Args:
            predictions: List of K tensors, each (N, input_dim).
            targets: List of K tensors, each (N, input_dim).
        """
        K = len(predictions)
        total = torch.tensor(0.0, device=predictions[0].device)
        for pred, tgt in zip(predictions, targets):
            total = total + torch.mean((pred - tgt) ** 2)
        return total / K

    def topological_preservation_loss(
        self,
        sheaf_laplacians: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        L_topo = (1/|E|) * sum_e max(0, Delta_e(t) - tau)

        Hinge penalty on edge weight changes exceeding threshold tau per step.
        Analogous to Fused Lasso in GGANO.

        Measures the change in off-diagonal Laplacian blocks between consecutive
        time steps, penalizing rapid topological rewiring.

        Args:
            sheaf_laplacians: List of K tensors, each (N*d, N*d).
        """
        K = len(sheaf_laplacians)
        if K < 2:
            return torch.tensor(0.0, device=sheaf_laplacians[0].device)

        total = torch.tensor(0.0, device=sheaf_laplacians[0].device)
        count = 0
        for t in range(K - 1):
            # Change in Laplacian between consecutive steps
            delta = sheaf_laplacians[t + 1] - sheaf_laplacians[t]

            # Per-element hinge: penalize changes exceeding tau
            hinge = torch.relu(delta.abs() - self.tau)
            total = total + hinge.mean()
            count += 1

        return total / max(count, 1)

    def regulon_activity_loss(
        self,
        predicted_activity: torch.Tensor,
        viper_activity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_regulon = -(1/N) * sum_v rho(a_hat_v, a_VIPER_v)
 
        Negative correlation between predicted and VIPER TF activity.
        Uses differentiable NeuralSort-based Spearman approximation.
 
        Args:
            predicted_activity: (K, N) predicted TF activity.
            viper_activity: (K, N) VIPER-inferred activity.
            mask: Optional (N,) boolean mask to restrict supervision to high-confidence TFs.
        """
        if viper_activity is None or viper_activity.sum() == 0:
            return torch.tensor(0.0, device=predicted_activity.device)
 
        pa = predicted_activity
        va = viper_activity
        if mask is not None:
            # Only supervise masked TFs (break circular logic)
            pa = pa[:, mask]
            va = va[:, mask]
 
        # Use differentiable Spearman via NeuralSort ranking
        corr = _differentiable_spearman(
            pa, va, temperature=self.neuralsort_temperature
        )
        return -corr  # Negative because we want to maximize correlation

    def sparsity_loss(
        self,
        sheaf_laplacians: List[torch.Tensor],
        allele_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_sparse = ||dL_F / dz_allele||_1

        L1 penalty on allele-specific perturbation of the sheaf Laplacian.
        Enforces k-hop locality: rewiring should be sparse near TP53 node.

        Uses torch.autograd.grad with create_graph=False to compute the L1
        norm of the gradient without building a second-order graph. This saves
        memory at the cost of not backpropping through the gradient itself,
        which is acceptable for a regulariser.

        Args:
            sheaf_laplacians: List of K tensors, each (N*d, N*d).
            allele_embedding: (esm_dim,) or (1, esm_dim) allele embedding.
        """
        if not allele_embedding.requires_grad:
            return torch.tensor(0.0, device=allele_embedding.device)

        # Sum all Laplacian entries (scalar) to compute gradient
        L_sum = sum(L.sum() for L in sheaf_laplacians)

        # Compute gradient of L_F w.r.t. allele embedding.
        # create_graph=True is REQUIRED to backprop through the gradient (second-order).
        # This allows the model to learn to be sparse w.r.t. the allele conditioning.
        # WARNING: This is memory intensive.
        grad = torch.autograd.grad(
            L_sum,
            allele_embedding,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        if grad[0] is None or not torch.isfinite(grad[0]).all():
            return torch.tensor(0.0, device=allele_embedding.device)

        # L1 norm of the gradient
        return grad[0].abs().mean()

    def velocity_consistency_loss(
        self,
        predictions: List[torch.Tensor],
        velocity_fields: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        L_vel = -mean cosine_similarity(X_hat(t+1) - X_hat(t), v(t))

        Penalises predicted trajectories that contradict RNA velocity direction.
        Only active when velocity_fields is provided (from scVelo).

        Args:
            predictions: List of K tensors (N, input_dim).
            velocity_fields: Optional list of K-1 tensors (N, input_dim) from scVelo.
        """
        if velocity_fields is None or len(velocity_fields) == 0:
            return torch.tensor(0.0, device=predictions[0].device)

        import torch.nn.functional as F
        total = torch.tensor(0.0, device=predictions[0].device)
        count = 0
        for t in range(min(len(predictions) - 1, len(velocity_fields))):
            delta = predictions[t + 1] - predictions[t]   # (N, genes)
            vel   = velocity_fields[t]                     # (N, genes)
            cos   = F.cosine_similarity(delta, vel, dim=-1)  # (N,)
            total = total - cos.mean()
            count += 1
        return total / max(count, 1)

    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        sheaf_laplacians: List[torch.Tensor],
        predicted_activity: Optional[torch.Tensor] = None,
        viper_activity: Optional[torch.Tensor] = None,
        regulon_mask: Optional[torch.Tensor] = None,
        allele_embedding: Optional[torch.Tensor] = None,
        velocity_fields: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.

        Design rules
        ------------
        1. L_expr is ALWAYS computed (it's the primary signal).
        2. All other terms are SKIPPED when their lambda is 0. This avoids the
           `0 * NaN = NaN` trap that poisons the total loss when a disabled
           regulariser produces NaN during curriculum warm-up.
        3. Every term is passed through `nan_to_num` before entering the sum,
           so a single bad batch can never poison the optimizer step.

        Returns:
            total_loss: Scalar loss for backpropagation.
            components: Dict of individual loss component values (detached).
        """
        # ── L_expr: always active ────────────────────────────────────────
        l_expr = self.expression_loss(predictions, targets)
        device = l_expr.device

        _zero = torch.tensor(0.0, device=device)

        def _safe(t: torch.Tensor) -> torch.Tensor:
            """Replace NaN/Inf with 0 so a bad batch never poisons the total."""
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        # ── L_topo: only if lambda_1 > 0 ─────────────────────────────────
        if self.lambda_1 > 0.0:
            l_topo = self.topological_preservation_loss(sheaf_laplacians)
        else:
            l_topo = _zero

        # ── L_regulon: only if lambda_2 > 0 and VIPER provided ───────────
        if (
            self.lambda_2 > 0.0
            and predicted_activity is not None
            and viper_activity is not None
        ):
            l_regulon = self.regulon_activity_loss(
                predicted_activity, viper_activity, mask=regulon_mask
            )
        else:
            l_regulon = _zero

        # ── L_sparse: only if lambda_3 > 0 and allele emb has grad ───────
        # CRITICAL: sparsity_loss calls torch.autograd.grad through the full
        # forward graph. Under fp16 autocast this produces NaN. Only run it
        # when it will actually contribute to the loss.
        if (
            self.lambda_3 > 0.0
            and allele_embedding is not None
            and allele_embedding.requires_grad
        ):
            l_sparse = self.sparsity_loss(sheaf_laplacians, allele_embedding)
        else:
            l_sparse = _zero

        # ── L_vel: only if lambda_4 > 0 and velocity fields provided ─────
        if self.lambda_4 > 0.0 and velocity_fields is not None and len(velocity_fields) > 0:
            l_vel = self.velocity_consistency_loss(predictions, velocity_fields)
        else:
            l_vel = _zero

        # ── Total: every term sanitised ──────────────────────────────────
        total = (
            _safe(l_expr)
            + self.lambda_1 * _safe(l_topo)
            + self.lambda_2 * _safe(l_regulon)
            + self.lambda_3 * _safe(l_sparse)
            + self.lambda_4 * _safe(l_vel)
        )

        components = {
            "loss_expr": float(l_expr.detach().item()),
            "loss_topo": float(l_topo.detach().item()) if torch.is_tensor(l_topo) else 0.0,
            "loss_regulon": float(l_regulon.detach().item()) if torch.is_tensor(l_regulon) else 0.0,
            "loss_sparse": float(l_sparse.detach().item()) if torch.is_tensor(l_sparse) else 0.0,
            "loss_vel": float(l_vel.detach().item()) if torch.is_tensor(l_vel) else 0.0,
            "loss_total": float(total.detach().item()),
        }

        # Debug: log which component produced NaN so the user sees the culprit
        if not torch.isfinite(total):
            bad = [k for k, v in components.items()
                   if k != "loss_total" and not torch.isfinite(torch.tensor(v))]
            logger.warning(
                f"Non-finite total loss. Components={components}. Bad terms={bad}"
            )

        return total, components


def _neural_sort_ranks(x: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    Differentiable soft ranking via NeuralSort (Grover et al., NeurIPS 2019).

    For each element j with score s[j], the logit for element j having rank k is:

        logit[k, j] = (n + 1 - 2k) * s[j] - sum_l |s[j] - s[l]|

    A column-wise softmax over the rank dimension k (scaled by tau) gives a
    soft permutation matrix P where P[k, j] is the probability that element j
    has rank k. The expected rank is then:

        soft_rank[j] = sum_k  k * P[k, j]

    Args:
        x: (..., N) input scores.
        tau: Temperature for softmax (lower = closer to true rank).

    Returns:
        soft_ranks: (..., N) differentiable rank estimates in [1, N].
    """
    # x: (..., N)
    n = x.shape[-1]
    device = x.device
    dtype = x.dtype

    # Absolute pairwise differences: (..., N, N)  abs_diff[..., j, l] = |s[j] - s[l]|
    x_i = x.unsqueeze(-1)   # (..., N, 1)
    x_j = x.unsqueeze(-2)   # (..., 1, N)
    abs_diff = (x_i - x_j).abs()           # (..., N, N)
    sum_abs_diff = abs_diff.sum(dim=-1)     # (..., N)  sum_l |s[j] - s[l]|

    # Rank indices k = 1 .. n, shape (n,)
    k = torch.arange(1, n + 1, device=device, dtype=dtype)  # (n,)

    # Broadcast to (..., n, N):
    #   logit[..., k-1, j] = (n + 1 - 2k) * s[j] - sum_l |s[j] - s[l]|
    # x: (..., N)  ->  (..., 1, N)
    # k: (n,)      ->  (n, 1)
    coeff = (n + 1 - 2 * k).unsqueeze(-1)       # (n, 1)
    x_bc = x.unsqueeze(-2)                       # (..., 1, N)
    sad_bc = sum_abs_diff.unsqueeze(-2)          # (..., 1, N)

    logits = coeff * x_bc - sad_bc               # (..., n, N)

    # Softmax over ELEMENT dimension (dim=-1) for each rank row k.
    # P[..., k, j] = prob that element j is the k-th largest.
    # Row k of P is a distribution over which element holds rank k.
    P = torch.softmax(logits / tau, dim=-1)      # (..., n, N)

    # Expected rank: sum_k k * P[k, j]
    k_bc = k.view(*([1] * (x.dim() - 1)), n, 1)  # (..., n, 1)  -- broadcast over N
    soft_ranks = (k_bc * P).sum(dim=-2)           # (..., N)

    return soft_ranks


# Backward-compatible alias so existing tests that import _soft_rank continue to work.
_soft_rank = _neural_sort_ranks


def _differentiable_spearman(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable Spearman correlation via NeuralSort soft ranking.

    Standard Spearman uses hard ranking (non-differentiable).
    We approximate ranks using NeuralSort (Grover et al., NeurIPS 2019),
    then compute Pearson correlation on the resulting soft ranks.

    IMPORTANT: We normalize inputs to zero mean, unit variance BEFORE ranking.
    This decouples the temperature from the scale of the inputs, making
    tau=1.0 universally appropriate regardless of activity magnitude.

    Args:
        pred: (..., N) predictions.
        target: (..., N) targets.
        temperature: Softness parameter (lower = closer to true rank).

    Returns:
        Mean Spearman correlation (scalar).
    """
    # Normalize to zero mean, unit variance (scale-invariant ranking)
    def _normalize(x):
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mu) / std

    pred_normed = _normalize(pred)
    target_normed = _normalize(target)

    # NeuralSort soft ranking
    pred_ranks = _neural_sort_ranks(pred_normed, tau=temperature)
    target_ranks = _neural_sort_ranks(target_normed, tau=temperature)

    # Pearson on soft ranks = Spearman approximation
    return _pearson_correlation(pred_ranks, target_ranks)


def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched Pearson correlation along the last dimension.

    Numerically defended against fp16 round-off:
    - Variance is clamped BEFORE sqrt (fp16 can round a tiny positive value
      below zero, and sqrt(-eps) = NaN).
    - Denominator is clamped to prevent division by a subnormal.
    - Final correlation is clamped to [-1, 1] to catch any residual drift.
    """
    x_mean = x - x.mean(dim=-1, keepdim=True)
    y_mean = y - y.mean(dim=-1, keepdim=True)

    cov = (x_mean * y_mean).mean(dim=-1)
    var_x = x_mean.pow(2).mean(dim=-1).clamp(min=1e-8)
    var_y = y_mean.pow(2).mean(dim=-1).clamp(min=1e-8)
    std_x = var_x.sqrt()
    std_y = var_y.sqrt()

    denom = (std_x * std_y).clamp(min=1e-8)
    corr = (cov / denom).clamp(min=-1.0, max=1.0)
    return corr.mean()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    K, N, d, input_dim = 10, 50, 4, 50
    Nd = N * d

    # Synthetic data — Laplacians must be in computation graph for sparsity loss
    predictions = [torch.randn(N, input_dim) for _ in range(K)]
    targets = [torch.randn(N, input_dim) for _ in range(K)]
    allele_emb = torch.randn(1280, requires_grad=True)
    # Create Laplacians that depend on allele_emb (simulating real forward pass)
    base_L = torch.randn(Nd, Nd)
    laplacians = [base_L + allele_emb.sum() * 0.001 * torch.randn(Nd, Nd) for _ in range(K)]
    pred_activity = torch.randn(K, N)
    viper_activity = torch.randn(K, N)

    criterion = TSGNNLoss(lambda_1=0.1, lambda_2=0.5, lambda_3=0.01, lambda_4=0.1, tau=0.5)

    # Test with all components
    total, components = criterion(
        predictions, targets, laplacians,
        pred_activity, viper_activity, allele_emb,
    )

    print("Loss components:")
    for name, val in components.items():
        print(f"  {name}: {val:.6f}")

    # Test backward
    total.backward()
    print(f"\nAllele embedding gradient norm: {allele_emb.grad.norm():.6f}")

    # Test without optional components
    total2, comp2 = criterion(predictions, targets, laplacians)
    print(f"\nWithout optional losses: {comp2}")

    # Test with velocity fields
    vel_fields = [torch.randn(N, input_dim) for _ in range(K - 1)]
    total3, comp3 = criterion(predictions, targets, laplacians, velocity_fields=vel_fields)
    print(f"\nWith velocity fields: {comp3}")
