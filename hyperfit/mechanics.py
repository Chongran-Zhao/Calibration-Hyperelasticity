"""Continuum-mechanics primitives shared across the package.

Deformation-gradient construction for the supported loading modes, extraction
of the observable stress components per mode, and the inverse-Langevin
approximation used by micromechanical models.
"""

from __future__ import annotations

import numpy as np

#: Loading modes with a diagonal deformation gradient.
DIAGONAL_MODES = ("UT", "UC", "ET", "PS", "BT")

#: Loading modes observed through the shear component P12.
SHEAR_MODES = ("SS", "CSS")


def get_deformation_gradient(stretch, mode):
    """Construct the deformation gradient ``F`` for a loading mode.

    Args:
        stretch: Scalar stretch/shear, or a pair for ``BT`` (lambda1, lambda2)
            and ``CSS`` (gamma, lambda1).
        mode: One of ``UT, UC, ET, PS, SS, CSS, BT``.
    """
    if mode in ("UT", "UC"):
        # Uniaxial: diag(lambda, lambda^-1/2, lambda^-1/2)
        return np.diag([stretch, stretch**-0.5, stretch**-0.5])
    if mode == "ET":
        # Equibiaxial: diag(lambda, lambda, lambda^-2)
        return np.diag([stretch, stretch, stretch**-2.0])
    if mode == "PS":
        # Pure shear: diag(lambda, 1, lambda^-1)
        return np.diag([stretch, 1.0, stretch**-1.0])
    if mode == "SS":
        # Simple shear: gamma in the x-y plane
        return np.array([
            [1.0, stretch, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    if mode == "CSS":
        # Compound simple shear: axial stretch lambda1 plus shear gamma
        if isinstance(stretch, (tuple, list, np.ndarray)) and len(stretch) == 2:
            gamma, lam1 = float(stretch[0]), float(stretch[1])
        else:
            gamma, lam1 = float(stretch), 1.0
        lam2 = lam1**-0.5
        return np.array([
            [lam1, gamma, 0.0],
            [0.0, lam2, 0.0],
            [0.0, 0.0, lam2],
        ])
    if mode == "BT":
        # Biaxial: diag(lambda1, lambda2, (lambda1*lambda2)^-1)
        lam1, lam2 = stretch
        return np.diag([lam1, lam2, (lam1 * lam2)**-1.0])
    raise ValueError(f"Unsupported mode: {mode}")


def get_stress_components(stress, mode):
    """Extract the observable stress components of a single tensor.

    Returns a list ordered as the experiment reports them: ``[P11]`` for
    uniaxial/equibiaxial, ``[P11, P22]`` for pure-shear/biaxial, ``[P12]``
    for the shear modes.
    """
    if mode in ("UT", "UC", "ET"):
        return [stress[0, 0]]
    if mode in ("PS", "BT"):
        return [stress[0, 0], stress[1, 1]]
    if mode in ("SS", "CSS"):
        return [stress[0, 1]]
    raise ValueError(f"Unsupported mode: {mode}")


def stress_components_batch(stress, mode):
    """Vectorised :func:`get_stress_components` for a stack of tensors.

    Args:
        stress: Array of shape ``(n, 3, 3)``.

    Returns:
        Tuple of 1-D arrays, one per observable component.
    """
    if mode in ("UT", "UC", "ET"):
        return (stress[:, 0, 0],)
    if mode in ("PS", "BT"):
        return (stress[:, 0, 0], stress[:, 1, 1])
    if mode in ("SS", "CSS"):
        return (stress[:, 0, 1],)
    raise ValueError(f"Unsupported mode: {mode}")


def inv_Langevin_Kroger(x):
    """Inverse Langevin approximation, Kroger (JNNFM 2015), Eqn. 14.

    Works on floats, numpy arrays and sympy symbols alike.
    """
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    top = 15.0 - (6.0 * x2 + x4 - 2.0 * x6)
    bot = 5.0 * (1.0 - x2)
    return x * top / bot
