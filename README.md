# Hyperelastic Material Calibration

Desktop and web workbench for fitting hyperelastic material models to
experimental data and running prediction workflows. The computational core is
the `hyperfit` Python package; the UI is a React frontend served by a FastAPI
backend, either inside a native desktop window or in the browser.

## Install

```bash
git clone https://github.com/Chongran-Zhao/Calibration-Hyperelasticity.git
cd Calibration-Hyperelasticity
pip install -r requirements.txt
```

## Run the Desktop App

```bash
python3 -m hyperfit.desktop
```

This starts a local server on a free port and opens the workbench in a native
window. Variants:

```bash
python3 -m hyperfit.desktop --browser   # serve and open the default browser
python3 -m hyperfit.desktop --check     # headless smoke test
```

Installing the package (`pip install -e .`) also provides the `hyperfit-app`
command, which is the same launcher.

The desktop app serves the prebuilt frontend from `frontend/dist`. After
changing frontend code, rebuild it with `cd frontend && npm run build`.

## Frontend Development

For hot-reloading UI work, run the API and the vite dev server separately:

```bash
uvicorn backend.main:app --reload        # API on :8000
cd frontend && npm install && npm run dev  # UI on :5173, /api proxied to :8000
```

## Project Structure

- `hyperfit/`: the Python package.
  - `models.py`, `strains.py`, `zhan.py`: material model registry, generalized
    strains, and the Zhan closed-form-stress models.
  - `kinematics.py`: PK1/PK2/Cauchy stress evaluation (scalar and vectorised
    batch paths) under incompressibility.
  - `mechanics.py`, `evaluation.py`: loading-mode geometry and the single
    source of truth mapping stress tensors to experimental observables.
  - `datasets.py`: HDF5/text experimental data loading.
  - `optimizer.py`: R²-normalized calibration on scipy solvers.
  - `network.py`: parallel (isostrain) composition of models.
  - `plotting.py`: matplotlib comparison plots.
  - `api/`: FastAPI layer (routes, presentation metadata, services).
  - `desktop.py`: desktop launcher (uvicorn + pywebview window).
- `backend/main.py`: compatibility shim exposing `hyperfit.api` for uvicorn.
- `frontend/`: React/Tailwind web interface (vite).
- `data/data.h5`: packaged experimental datasets.
- `tests/`: numerical regression suite (`python3 tests/test_regression.py`)
  pinning results to the pre-refactor baseline.
- `assets/examples/`: example screenshots and animations.
- `DESIGN.md`: design direction for the web interface.

## Testing

```bash
python3 tests/test_regression.py
```

Recomputes stress tensors, objective values and full calibrations for every
model family against `tests/baseline.json` and fails on numerical drift.

## Example: Zhan (Non-Gaussian)

This example reproduces Fig. 7 from Zhan (JMPS) by fitting the Zhan
(non-Gaussian) model to James 1975 uniaxial tension data, then predicting
biaxial tension.

![Example](assets/examples/example.gif)
