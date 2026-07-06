"""Compatibility shim: the API now lives in :mod:`hyperfit.api`.

Kept so the documented ``uvicorn backend.main:app --reload`` keeps working.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperfit.api import create_app

app = create_app()
