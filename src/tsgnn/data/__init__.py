"""TS-GNN Data Loading and Preprocessing."""

import os
from pathlib import Path


def _resolve_data_dir() -> Path:
    """Resolve the project data directory with a four-level fallback.

    Priority:
      1. ``TSGNN_DATA_DIR`` environment variable (production / custom deployments).
      2. A sibling ``data/`` directory of the package repository (editable install
         layout: ``<repo>/src/tsgnn/data/__init__.py`` → ``<repo>/data``).
      3. A ``data/`` directory next to the package when installed via wheel
         (``<site-packages>/tsgnn/data/__init__.py`` → ``<site-packages>/data``).
      4. Current working directory (Google Colab / ad-hoc runs).
    """
    env_dir = os.environ.get("TSGNN_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    here = Path(__file__).resolve()
    # Walk up from tsgnn/data/__init__.py looking for a sibling "data" folder.
    # parents[2] == src/, parents[3] == repo root when editable; parents[1] == tsgnn/
    # for wheel installs. Accept either layout without crashing.
    for depth in (3, 2, 1):
        try:
            candidate = here.parents[depth] / "data"
        except IndexError:
            continue
        if candidate.exists():
            return candidate

    # Final fallback: cwd-local data folder. Always returns a path — never raises.
    return Path(os.getcwd()) / "data"


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"

__all__ = [
    "download", "preprocess", "allele", "grn_construction", "temporal",
    "brca_loader", "biokey_loader", "crc_loader", "metabric_loader",
    "multiprior_grn", "registry",
    "DATA_DIR", "RAW_DIR", "EXTERNAL_DIR", "_resolve_data_dir",
    # Download helpers (also importable as tsgnn.data.download_biokey_scrna etc.)
    "download_biokey_scrna", "download_metabric", "download_dorothea",
    "download_crc_scrna",
    # Dataset loaders (lazy-resolved via __getattr__)
    "load_brca_data", "load_biokey_data", "load_crc_data", "load_metabric_data",
    "biokey_available", "crc_available", "metabric_available",
    "evaluate_biokey_holdout",
    "build_multiprior_grn",
    "load_dataset",
    "intersect_gene_sets",
]

# Lazy re-exports so `from tsgnn.data import download_biokey_scrna` works
# without paying the cost of importing the full download module on package init.
def __getattr__(name: str):
    _download_extras = {
        "download_biokey_scrna", "download_metabric",
        "download_dorothea", "download_crc_scrna",
    }
    if name in _download_extras:
        from tsgnn.data import download as _dl
        return getattr(_dl, name)

    _loader_extras = {
        "load_brca_data": ("brca_loader", "load_brca_data"),
        "load_biokey_data": ("biokey_loader", "load_biokey_data"),
        "biokey_available": ("biokey_loader", "biokey_available"),
        "evaluate_biokey_holdout": ("biokey_loader", "evaluate_biokey_holdout"),
        "load_crc_data": ("crc_loader", "load_crc_data"),
        "crc_available": ("crc_loader", "crc_available"),
        "load_metabric_data": ("metabric_loader", "load_metabric_data"),
        "metabric_available": ("metabric_loader", "metabric_available"),
        "build_multiprior_grn": ("multiprior_grn", "build_multiprior_grn"),
        "load_dataset": ("registry", "load_dataset"),
        "intersect_gene_sets": ("nature_adapter", "intersect_gene_sets"),
    }
    if name in _loader_extras:
        mod, attr = _loader_extras[name]
        import importlib
        m = importlib.import_module(f"tsgnn.data.{mod}")
        return getattr(m, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
