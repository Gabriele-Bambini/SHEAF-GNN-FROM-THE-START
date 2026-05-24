"""
Unified dataset registry.

Single entry point for every benchmark dataset used in the paper:

    from tsgnn.data import load_dataset
    a = load_dataset("brca_wu2021")        # single-cell BRCA
    b = load_dataset("biokey", malignant_only=True)
    c = load_dataset("crc_pelka2021")
    d = load_dataset("metabric", hotspots_only=True)

The registry normalizes the return type to `AnnData` (except where a raw
sparse matrix is intentionally returned). This lets the training / evaluation
pipeline depend on a single abstraction.
"""
from __future__ import annotations

from typing import Any

_REGISTRY = {
    "brca_wu2021": ("brca_loader", "load_brca_data"),
    "biokey": ("biokey_loader", "load_biokey_data"),
    "crc_pelka2021": ("crc_loader", "load_crc_data"),
    "metabric": ("metabric_loader", "load_metabric_data"),
    "ovarian_izar2020": ("ovarian_loader", "load_ovarian_data"),
}

ALIASES = {
    "brca": "brca_wu2021", "wu2021": "brca_wu2021",
    "crc": "crc_pelka2021", "pelka2021": "crc_pelka2021",
    "e-mtab-8107": "biokey", "bassez2021": "biokey",
    "ovarian": "ovarian_izar2020", "izar2020": "ovarian_izar2020",
    "hgsoc": "ovarian_izar2020",
}


def list_datasets() -> list:
    return sorted(_REGISTRY)


def load_dataset(name: str, **kwargs: Any):
    key = ALIASES.get(name.lower(), name.lower())
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Known: {list_datasets()}")
    import importlib
    mod_name, attr = _REGISTRY[key]
    m = importlib.import_module(f"tsgnn.data.{mod_name}")
    return getattr(m, attr)(**kwargs)
