from __future__ import annotations

import os


def apply_platform_fixes() -> None:
    """
    Apply small runtime compatibility tweaks.

    Windows:
    - Work around Conda/OpenMP duplicate runtime errors that can prevent
      importing PyTorch (OMP Error #15) by setting KMP_DUPLICATE_LIB_OK.
    """

    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

