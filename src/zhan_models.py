import os
import numpy as np

from utils import inv_Langevin_Kroger

_LEBEDEV_CACHE = None


def _load_lebedev():
    global _LEBEDEV_CACHE
    if _LEBEDEV_CACHE is not None:
        return _LEBEDEV_CACHE
    data_path = os.path.join(os.path.dirname(__file__), "lebedev", "Lebedev.txt")
    data = np.loadtxt(data_path)
    phi = data[:, 0]
    theta = data[:, 1]
    weight = data[:, 2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    _LEBEDEV_CACHE = (cos_t, sin_t, cos_p, sin_p, weight)
    return _LEBEDEV_CACHE


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


def zhan_gaussian_pk1(F, params):
    mu = params["mu"]
    C = F.T @ F
    lambdas, N1, N2, N3 = _eigen_decomp(C)
    lambda_1, lambda_2, lambda_3 = lambdas
    J = lambda_1 * lambda_2 * lambda_3
    tilde_lambda_1 = J ** (-1.0 / 3.0) * lambda_1
    tilde_lambda_2 = J ** (-1.0 / 3.0) * lambda_2
    tilde_lambda_3 = J ** (-1.0 / 3.0) * lambda_3

    scale = (tilde_lambda_1 + tilde_lambda_2 + tilde_lambda_3)
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
