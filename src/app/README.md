# Desktop GUI Refactor Map

This package is the landing area for the PySide desktop app refactor.

- `catalog.py`: Stable labels, references, display names, and dataset/model metadata.
- `formatting.py`: Domain label helpers for plots, stress components, loading modes, and axes.
- `theme.py`: Palette, stylesheet, font, and plot-theme helpers.

Planned next modules:

- `widgets/`: Reusable controls such as dataset pickers, parameter tables, spring builders, and plot canvases.
- `pages/`: Data, Model, Fit, and Predict workspaces.
- `state.py`: UI state and stale-result tracking.
- `workers.py`: Optimization and background-task workers.
