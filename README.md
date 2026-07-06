<div align="center">

<img src="frontend/public/logo.svg" width="104" height="104" alt="Calibration for Hyperelasticity logo" />

# Calibration for Hyperelasticity

**A desktop &amp; web workbench for fitting hyperelastic material models to experimental stress–stretch data — and predicting new loading modes.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19-149ECA?logo=react&logoColor=white)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2563EB.svg)](#license)
[![Platform](https://img.shields.io/badge/Desktop-macOS%20%C2%B7%20Web-0E1B33.svg)](#run-the-desktop-app)

</div>

<div align="center">
  <img src="assets/screenshots/workbench.png" width="880" alt="The calibration workbench: experimental data selection, live stress–stretch preview, and the four-step workflow" />
</div>

---

## Overview

**Calibration for Hyperelasticity** turns a pile of experimental stress–stretch
curves into calibrated constitutive models. Pick a dataset, compose a
strain-energy model (single or a parallel network of springs), run the
optimizer, and reuse the fitted parameters to predict loading modes you never
measured — all through a guided four-step workflow.

The computational core is the **`hyperfit`** Python package. The interface is a
React frontend served by FastAPI, running either **inside a native desktop
window** or in the browser. The two share one codebase and one server.

```
Experimental Data  →  Model Architecture  →  Optimization  →  Prediction
```

## Highlights

- **Guided workflow** — a four-step pipeline from raw data to prediction, with a
  numbered progress rail so you always know where you are.
- **Rich model library** — Neo-Hookean, Mooney–Rivlin, Yeoh, Arruda–Boyce,
  Ogden (*n*-term), the Zhan Gaussian / non-Gaussian micromechanical models,
  and a configurable Hill family over five generalized strain measures.
- **Parallel networks** — compose models as isostrain springs; parameters are
  namespaced and fit jointly.
- **Vectorized core** — stress evaluation runs on batched NumPy, giving a
  median **~7× speedup** on objective evaluations (up to 150× on large sets)
  versus the original implementation.
- **8 built-in datasets** — 55 loading configurations across rubber, elastomer,
  gel and brain-tissue experiments, packaged in a single HDF5 file.
- **Reproducible** — a numerical regression suite pins every model's stress
  tensors, objective values and full calibrations to a recorded baseline.

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
python3 -m hyperfit.desktop --check     # headless smoke test (start, probe, exit)
```

Installing the package (`pip install -e .`) also provides the **`hyperfit-app`**
command — the same launcher. The app serves the prebuilt frontend from
`frontend/dist`; after changing UI code, rebuild with `cd frontend && npm run build`.

> **macOS + system Python 3.9:** the native window needs `pywebview`, which
> pulls in `pyobjc`. On 3.9 the requirements pin `pyobjc<12` (newer releases
> ship wheels only for 3.10+). `--browser` mode has no such dependency.

## Frontend Development

For hot-reloading UI work, run the API and the vite dev server separately:

```bash
uvicorn backend.main:app --reload          # API on :8000
cd frontend && npm install && npm run dev   # UI on :5173, /api proxied to :8000
```

## Model Library

| Model | Type | Parameters |
| --- | --- | --- |
| Neo-Hookean | Invariant | `C₁` |
| Mooney–Rivlin | Invariant | `C₁, C₂` |
| Yeoh | Invariant | `C₁, C₂, C₃` |
| Arruda–Boyce | Invariant (micromechanical) | `μ, N` |
| Ogden (*n*-term) | Stretch | `μᵢ, αᵢ` |
| Modified Ogden | Stretch | `μ, α` |
| Hill (5 strain measures) | Stretch | `μ, mᵢ, nᵢ` |
| Zhan Gaussian | Closed-form PK1 | `μ` |
| Zhan non-Gaussian | Closed-form PK1 (Lebedev) | `μ, N` |

Any subset can be combined into a **parallel network**. Generalized strains for
the Hill family: Seth–Hill, Hencky, Curnier–Rakotomanana, Curnier–Zysset,
Darijani–Naghdabadi.

## Datasets

Eight experimental sources ship in `data/data.h5` (55 loading configurations):

| Source | Material |
| --- | --- |
| Treloar (1944) | Vulcanized rubber |
| James, Green &amp; Simpson (1975) | Rubber |
| Jones &amp; Treloar (1975) | Rubber, biaxial |
| Kawabata et al. (1981) | Rubber, biaxial |
| Kawamura et al. (2001) | Elastomer |
| Katashima et al. (2012) | Polymer gel |
| Meunier et al. (2008) | Silicone rubber |
| Budday et al. (2017) | Human brain tissue |

Supported loading modes: uniaxial tension/compression (UT/UC), equibiaxial
tension (ET), pure shear (PS), simple &amp; compound simple shear (SS/CSS),
biaxial tension (BT).

## Project Structure

```
hyperfit/                 the calibration core (Python package)
├── models.py             model registry + Ogden / Hill factories
├── strains.py            generalized strain measures
├── zhan.py               Zhan closed-form-stress models (scalar + batch)
├── kinematics.py         PK1/PK2/Cauchy stress (scalar + vectorized batch)
├── mechanics.py          loading-mode geometry, inverse Langevin
├── evaluation.py         single source of truth: tensors → observables
├── datasets.py           HDF5 / text experimental-data loading
├── optimizer.py          R²-normalized calibration on SciPy solvers
├── network.py            parallel (isostrain) model composition
├── plotting.py           matplotlib comparison plots
├── desktop.py            desktop launcher (uvicorn + pywebview)
└── api/                  FastAPI layer (routes · metadata · services)

backend/main.py           compatibility shim exposing hyperfit.api
frontend/                 React / Tailwind web interface (vite)
data/data.h5              packaged experimental datasets
tests/                    numerical regression suite + baseline
```

## Testing

```bash
python3 tests/test_regression.py
```

Recomputes stress tensors, objective values and full calibrations for every
model family against `tests/baseline.json` and fails on numerical drift.

## Example: Zhan non-Gaussian

The signature example reproduces Fig. 7 of Zhan et al. (JMPS) — fit the Zhan
non-Gaussian model to James (1975) uniaxial tension, then predict biaxial
tension:

<div align="center">
  <img src="assets/examples/example.gif" width="640" alt="Zhan non-Gaussian calibration and prediction" />
</div>

## Author

**Chongran Zhao** — Ph.D. student in Engineering, Brown University · M.Eng. in
Mechanics, SUSTech.
[Website](https://chongran-zhao.github.io) ·
[GitHub](https://github.com/Chongran-Zhao) ·
[Email](mailto:chongran_zhao@brown.edu)

## License

Released under the MIT License.
