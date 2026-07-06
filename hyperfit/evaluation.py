"""Mapping between model stress tensors and experimental observables.

A dataset dict (see :mod:`hyperfit.datasets`) declares, implicitly, which
scalar series the experiment reports: a single component, a component pair,
or the Cauchy stress difference sigma11 - sigma22. This module is the single
place that logic lives; the optimizer, the plotting helpers and the web API
all consume it.
"""

from __future__ import annotations

import numpy as np

from .mechanics import get_deformation_gradient, stress_components_batch


def experimental_values(dataset):
    """Observed series of a dataset, shaped ``(n,)`` or ``(n, k)``."""
    stress_exp = dataset["stress_exp"]
    bt_diff = dataset.get("bt_component") == "diff" and dataset["mode"] == "BT"
    if bt_diff and np.ndim(stress_exp) > 1:
        return stress_exp[:, 0] - stress_exp[:, 1]
    return stress_exp


def predict_stress_tensors(solver, dataset, params):
    """Model stress stack ``(n, 3, 3)`` in the dataset's stress measure."""
    F = np.asarray(dataset["F_list"], dtype=float)
    if dataset.get("stress_type", "PK1") == "cauchy":
        return solver.cauchy_batch(F, params)
    return solver.first_pk_batch(F, params)


def predict_observables(solver, dataset, params):
    """Model prediction of the dataset's observed series.

    Shape matches :func:`experimental_values` for the same dataset.
    """
    tensors = predict_stress_tensors(solver, dataset, params)
    comps = stress_components_batch(tensors, dataset["mode"])

    bt_diff = dataset.get("bt_component") == "diff" and dataset["mode"] == "BT"
    if bt_diff:
        return comps[0] - comps[1]

    component = dataset.get("component")
    if component == "22":
        return comps[1]
    if component == "11":
        return comps[0]

    if np.ndim(dataset["stress_exp"]) == 1:
        return comps[0]
    n_cols = dataset["stress_exp"].shape[1]
    return np.column_stack(comps[:n_cols])


def predict_curve(solver, mode, stretches, params, stress_type="PK1"):
    """Model observables on a synthetic stretch grid (for smooth curves).

    Args:
        stretches: 1-D array of stretch/shear values, or an iterable of
            ``(lam1, lam2)`` pairs for ``BT``.

    Returns:
        Tuple of 1-D arrays, one per observable component of the mode.
    """
    F = np.stack([get_deformation_gradient(s, mode) for s in stretches])
    if stress_type == "cauchy":
        tensors = solver.cauchy_batch(F, params)
    else:
        tensors = solver.first_pk_batch(F, params)
    return stress_components_batch(tensors, mode)
