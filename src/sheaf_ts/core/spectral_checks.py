"""Fail-closed spectral checks for sheaf Laplacians."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "SpectralReport",
    "check_normalized_spectrum",
    "check_psd",
    "check_symmetric",
    "dirichlet_energy",
]


@dataclass(frozen=True)
class SpectralReport:
    min_eigenvalue: float
    max_eigenvalue: float
    symmetry_error: float


def _require_square(matrix: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")


def check_symmetric(matrix: torch.Tensor, *, atol: float = 1e-8) -> float:
    """Raise if ``matrix`` is not symmetric; return max symmetry error."""
    _require_square(matrix)
    err = (matrix - matrix.T).abs().max().item()
    if err > atol:
        raise ValueError(f"matrix is not symmetric: max error {err:.3e}")
    return err


def check_psd(matrix: torch.Tensor, *, atol: float = 1e-8) -> SpectralReport:
    """Raise if ``matrix`` is not symmetric positive semidefinite."""
    symmetry_error = check_symmetric(matrix, atol=atol)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    min_eigenvalue = eigenvalues.min().item()
    max_eigenvalue = eigenvalues.max().item()
    if min_eigenvalue < -atol:
        raise ValueError(f"matrix is not PSD: min eigenvalue {min_eigenvalue:.3e}")
    return SpectralReport(
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
        symmetry_error=symmetry_error,
    )


def check_normalized_spectrum(
    matrix: torch.Tensor,
    *,
    atol: float = 1e-8,
) -> SpectralReport:
    """Raise unless a normalized Laplacian has spectrum inside ``[0, 2]``."""
    report = check_psd(matrix, atol=atol)
    if report.max_eigenvalue > 2.0 + atol:
        raise ValueError(
            "normalized Laplacian spectrum exceeds 2: "
            f"max eigenvalue {report.max_eigenvalue:.3e}"
        )
    return report


def dirichlet_energy(laplacian: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
    """Return ``0.5 * x.T @ L @ x`` for a flattened signal."""
    flat = signal.reshape(-1)
    if laplacian.shape != (flat.numel(), flat.numel()):
        raise ValueError(
            f"laplacian shape {tuple(laplacian.shape)} does not match "
            f"signal dimension {flat.numel()}"
        )
    return 0.5 * (flat @ laplacian @ flat)
