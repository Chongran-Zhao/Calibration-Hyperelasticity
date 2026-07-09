"""PyInstaller entry point for the macOS desktop bundle."""

import sys

from hyperfit.desktop import main


if __name__ == "__main__":
    sys.exit(main())
