"""Zhan et al. (JMPS 2023) micromechanical models with closed-form PK1 stress.

These models bypass the symbolic-energy pipeline: the first Piola-Kirchhoff
stress is evaluated directly, using a spectral decomposition and (for the
non-Gaussian variant) Lebedev quadrature over the unit sphere.

Each model ships in two forms: a scalar version operating on a single ``F``
and a ``_batch`` version operating on a stack ``(n, 3, 3)``. The batch
versions perform the identical floating-point operations element-wise, so
they reproduce the scalar results bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mechanics import inv_Langevin_Kroger

_LEBEDEV_FILE = Path(__file__).parent / "lebedev" / "Lebedev.txt"
_LEBEDEV_CACHE = None


def _load_lebedev():
    """Load and cache the Lebedev quadrature directions and weights."""
    global _LEBEDEV_CACHE
    if _LEBEDEV_CACHE is None:
        data = np.loadtxt(_LEBEDEV_FILE)
        phi, theta, weight = data[:, 0], data[:, 1], data[:, 2]
        _LEBEDEV_CACHE = (np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi), weight)
    return _LEBEDEV_CACHE


# --- Single-tensor helpers -------------------------------------------------

def _eigen_decomp(C):
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    lambdas = np.sqrt(vals)
    return lambdas, vecs[:, 0], vecs[:, 1], vecs[:, 2]


def _tensor_product(e1, e2, e3, N1, N2, N3):
    return e1 * np.outer(N1, N1) + e2 * np.outer(N2, N2) + e3 * np.outer(N3, N3)


def _dev(tensor, C):
    contraction = np.sum(tensor * C)
    return tensor - (1.0 / 3.0) * contraction * np.linalg.inv(C)


def _incompressible_constraint(P_ich, F):
    temp = np.linalg.inv(F).T
    pressure = P_ich[2, 2] / temp[2, 2]
    return -pressure * temp + P_ich


# --- Batch helpers ---------------------------------------------------------

def _eigen_decomp_batch(C):
    """Batched spectral decomposition; eigenvalues ascending as in eigh."""
    vals, vecs = np.linalg.eigh(C)
    return np.sqrt(vals), vecs


def _tensor_product_batch(e, vecs):
    """sum_k e_k N_k (x) N_k for stacks: e (n, 3), vecs (n, 3, 3) columns.

    Mirrors the scalar helper term by term so results match bit-for-bit.
    """
    def term(k):
        col = vecs[:, :, k]
        return e[:, k, None, None] * (col[:, :, None] * col[:, None, :])

    return term(0) + term(1) + term(2)


def _dev_batch(tensor, C):
    contraction = (tensor * C).reshape(tensor.shape[0], 9).sum(axis=1)
    return tensor - (1.0 / 3.0) * contraction[:, None, None] * np.linalg.inv(C)


def _incompressible_constraint_batch(P_ich, F):
    temp = np.linalg.inv(F).transpose(0, 2, 1)
    pressure = P_ich[:, 2, 2] / temp[:, 2, 2]
    return -pressure[:, None, None] * temp + P_ich


def _isochoric_stretches_batch(F):
    """C, sqrt-eigensystem and isochoric principal stretches for a stack."""
    C = np.matmul(F.transpose(0, 2, 1), F)
    lambdas, vecs = _eigen_decomp_batch(C)
    J = lambdas[:, 0] * lambdas[:, 1] * lambdas[:, 2]
    tilde = J[:, None] ** (-1.0 / 3.0) * lambdas
    return C, vecs, J, tilde


# --- Gaussian model ---------------------------------------------------------

def zhan_gaussian_pk1(F, params):
    mu = params["mu"]
    C = F.T @ F
    lambdas, N1, N2, N3 = _eigen_decomp(C)
    lambda_1, lambda_2, lambda_3 = lambdas
    J = lambda_1 * lambda_2 * lambda_3
    tilde_lambda_1 = J ** (-1.0 / 3.0) * lambda_1
    tilde_lambda_2 = J ** (-1.0 / 3.0) * lambda_2
    tilde_lambda_3 = J ** (-1.0 / 3.0) * lambda_3

    scale = tilde_lambda_1 + tilde_lambda_2 + tilde_lambda_3
    S_tilde = 2.0 * mu * (
        scale
        * _tensor_product(
            1.0 / tilde_lambda_1,
            1.0 / tilde_lambda_2,
            1.0 / tilde_lambda_3,
            N1,
            N2,
            N3,
        )
        + 2.0 * np.eye(3)
    )
    S_ich = J ** (-2.0 / 3.0) * _dev(S_tilde, C)
    P_ich = F @ S_ich
    return _incompressible_constraint(P_ich, F)


def zhan_gaussian_pk1_batch(F, params):
    mu = params["mu"]
    C, vecs, J, tilde = _isochoric_stretches_batch(F)

    scale = tilde[:, 0] + tilde[:, 1] + tilde[:, 2]
    S_tilde = 2.0 * mu * (
        scale[:, None, None] * _tensor_product_batch(1.0 / tilde, vecs)
        + 2.0 * np.eye(3)
    )
    S_ich = (J ** (-2.0 / 3.0))[:, None, None] * _dev_batch(S_tilde, C)
    P_ich = np.matmul(F, S_ich)
    return _incompressible_constraint_batch(P_ich, F)


# --- Non-Gaussian model ------------------------------------------------------

def zhan_nongaussian_pk1(F, params):
    mu = params["mu"]
    N = params["N"]

    C = F.T @ F
    lambdas, N1, N2, N3 = _eigen_decomp(C)
    lambda_1, lambda_2, lambda_3 = lambdas
    J = lambda_1 * lambda_2 * lambda_3

    tilde_lambda_1 = J ** (-1.0 / 3.0) * lambda_1
    tilde_lambda_2 = J ** (-1.0 / 3.0) * lambda_2
    tilde_lambda_3 = J ** (-1.0 / 3.0) * lambda_3

    cos_t, sin_t, cos_p, sin_p, weight = _load_lebedev()
    lam_i = (
        tilde_lambda_1 * cos_t**2
        + tilde_lambda_2 * sin_t**2 * cos_p**2
        + tilde_lambda_3 * sin_t**2 * sin_p**2
    )
    beta = inv_Langevin_Kroger(lam_i / np.sqrt(N))

    temp_1 = beta * cos_t**2
    temp_2 = beta * sin_t**2 * cos_p**2
    temp_3 = beta * sin_t**2 * sin_p**2

    S_tilde_1 = mu * np.sqrt(N) / tilde_lambda_1 * np.sum(temp_1 * weight)
    S_tilde_2 = mu * np.sqrt(N) / tilde_lambda_2 * np.sum(temp_2 * weight)
    S_tilde_3 = mu * np.sqrt(N) / tilde_lambda_3 * np.sum(temp_3 * weight)

    S_tilde = _tensor_product(S_tilde_1, S_tilde_2, S_tilde_3, N1, N2, N3)
    S_ich = J ** (-2.0 / 3.0) * _dev(S_tilde, C)
    P_ich = F @ S_ich
    return _incompressible_constraint(P_ich, F)


def zhan_nongaussian_pk1_batch(F, params):
    mu = params["mu"]
    N = params["N"]

    C, vecs, J, tilde = _isochoric_stretches_batch(F)

    cos_t, sin_t, cos_p, sin_p, weight = _load_lebedev()
    ct2, st2 = cos_t**2, sin_t**2
    cp2, sp2 = cos_p**2, sin_p**2

    # (n, q): mean chain stretch per quadrature direction. Multiplication
    # order matches the scalar version exactly (left-to-right).
    lam_i = (
        tilde[:, 0, None] * ct2
        + tilde[:, 1, None] * st2 * cp2
        + tilde[:, 2, None] * st2 * sp2
    )
    beta = inv_Langevin_Kroger(lam_i / np.sqrt(N))

    quad = np.stack(
        [
            np.sum(beta * ct2 * weight, axis=1),
            np.sum(beta * st2 * cp2 * weight, axis=1),
            np.sum(beta * st2 * sp2 * weight, axis=1),
        ],
        axis=1,
    )
    S_principal = mu * np.sqrt(N) / tilde * quad

    S_tilde = _tensor_product_batch(S_principal, vecs)
    S_ich = (J ** (-2.0 / 3.0))[:, None, None] * _dev_batch(S_tilde, C)
    P_ich = np.matmul(F, S_ich)
    return _incompressible_constraint_batch(P_ich, F)
