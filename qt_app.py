"""Backward-compatible entry point for the PySide desktop app.

The main implementation lives in ``desktop_app.py``.
"""

import multiprocessing

from desktop_app import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
