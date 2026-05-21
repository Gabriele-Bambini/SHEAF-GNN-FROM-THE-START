"""Auditable sheaf-math core."""

from sheaf_ts.core.coboundary import build_dense_coboundary
from sheaf_ts.core.laplacian_builders import (
    build_connection_laplacian,
    build_general_laplacian,
    normalize_block_laplacian,
)
from sheaf_ts.core.orthogonal import (
    cayley_map,
    matrix_exp_map,
    orthogonality_error,
    skew_from_params,
)
from sheaf_ts.core.spectral_checks import (
    SpectralReport,
    check_normalized_spectrum,
    check_psd,
    check_symmetric,
    dirichlet_energy,
)

__all__ = [
    "SpectralReport",
    "build_connection_laplacian",
    "build_dense_coboundary",
    "build_general_laplacian",
    "cayley_map",
    "check_normalized_spectrum",
    "check_psd",
    "check_symmetric",
    "dirichlet_energy",
    "matrix_exp_map",
    "normalize_block_laplacian",
    "orthogonality_error",
    "skew_from_params",
]
