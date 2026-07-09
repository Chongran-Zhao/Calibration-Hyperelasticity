"""Runtime paths shared by source checkouts and frozen desktop bundles."""

from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """Return the directory containing bundled ``data`` and ``frontend``."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent.parent
