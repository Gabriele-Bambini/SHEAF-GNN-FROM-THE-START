"""Clean sheaf time-series rebuild package.

This package is intentionally independent from the legacy ``tsgnn`` monolith.
The first stable surface is ``sheaf_ts.core``: small mathematical primitives
that can be tested without biology, ESM, scanpy, or HPC dependencies.
"""

__all__ = ["core"]
