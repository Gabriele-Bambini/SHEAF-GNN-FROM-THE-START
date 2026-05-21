"""TS-GNN Visualization Utilities.

Re-exports the most commonly used plotting functions so that users can write::

    from tsgnn.visualization import plot_rewiring_trajectory

instead of having to reach into the submodules. Imports are guarded so that
a missing optional dependency (e.g. matplotlib, networkx) does not prevent
the rest of the package from importing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Force a non-interactive backend when running headlessly (CI, HPC, Docker,
# Colab workers). Only set it if matplotlib has not already been imported,
# otherwise we leave the user's explicit choice in place. This makes figure
# generation safe without a $DISPLAY.
if "matplotlib" not in __import__("sys").modules and not os.environ.get("TSGNN_MPL_BACKEND_LEAVE"):
    try:  # pragma: no cover - trivial environment setup
        import matplotlib

        matplotlib.use("Agg", force=False)
    except ImportError:
        pass

_exports: list[str] = []

try:  # pragma: no cover - exercised via downstream scripts
    from .rewiring import plot_rewiring_trajectory  # noqa: F401
    _exports.append("plot_rewiring_trajectory")
except ImportError as e:
    logger.debug("rewiring visualisation unavailable: %s", e)

try:  # pragma: no cover
    from .attention import plot_attention_heatmap  # noqa: F401
    _exports.append("plot_attention_heatmap")
except ImportError as e:
    logger.debug("attention visualisation unavailable: %s", e)

try:  # pragma: no cover
    from .networks import plot_network  # noqa: F401
    _exports.append("plot_network")
except ImportError as e:
    logger.debug("networks visualisation unavailable: %s", e)

__all__ = _exports
