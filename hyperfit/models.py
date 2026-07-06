"""Hyperelastic material model registry.

Every model is a plain function tagged by :func:`register_model` with the
metadata the rest of the package consumes: model type (invariant / stretch /
custom), display formula, parameter names, initial guesses and bounds.
Factories generate parameterised families (multi-term Ogden, Hill with a
chosen generalized strain).
"""

from __future__ import annotations

import sympy as sp

from .mechanics import inv_Langevin_Kroger
from .strains import STRAIN_CONFIGS, STRAIN_FORMULAS, strain_function
from .zhan import (
    zhan_gaussian_pk1,
    zhan_gaussian_pk1_batch,
    zhan_nongaussian_pk1,
    zhan_nongaussian_pk1_batch,
)


def register_model(model_type, category, formula_str="", param_names=None, initial_guess=None, bounds=None):
    """Attach model metadata tags to an energy function."""

    def decorator(func):
        func.model_type = model_type
        func.category = category
        func.formula = formula_str
        func.param_names = param_names if param_names else []
        func.initial_guess = initial_guess if initial_guess else []
        func.bounds = bounds if bounds else []
        return func

    return decorator


class MaterialModels:
    """Unified collection of hyperelastic material models."""

    # --- Invariant based -----------------------------------------------------

    @staticmethod
    @register_model(
        model_type="invariant_based",
        category="phenomenological",
        formula_str=r"\Psi = C_1 (I_1 - 3)",
        param_names=["C1"],
        initial_guess=[0.5],
        bounds=[(1e-6, None)],  # C1 > 0
    )
    def NeoHookean(I1, params):
        """Neo-Hookean model."""
        return params["C1"] * (I1 - 3)

    @staticmethod
    @register_model(
        model_type="invariant_based",
        category="phenomenological",
        formula_str=r"\Psi = C_1 (I_1 - 3) + C_2 (I_2 - 3)",
        param_names=["C1", "C2"],
        initial_guess=[0.5, 0.1],
        bounds=[(1e-6, None), (None, None)],  # C1 > 0, C2 free
    )
    def MooneyRivlin(I1, I2, params):
        """Mooney-Rivlin model."""
        return params["C1"] * (I1 - 3) + params["C2"] * (I2 - 3)

    @staticmethod
    @register_model(
        model_type="invariant_based",
        category="phenomenological",
        formula_str=r"\Psi = C_1 (I_1 - 3) + C_2 (I_1 - 3)^2 + C_3 (I_1 - 3)^3",
        param_names=["C1", "C2", "C3"],
        initial_guess=[0.5, -0.01, 0.001],
        bounds=[(1e-6, None), (None, None), (None, None)],  # higher orders may go negative
    )
    def Yeoh(I1, params):
        """Yeoh model (3rd order)."""
        return (
            params["C1"] * (I1 - 3)
            + params["C2"] * (I1 - 3) ** 2
            + params["C3"] * (I1 - 3) ** 3
        )

    @staticmethod
    @register_model(
        model_type="invariant_based",
        category="micromechanical",
        formula_str=r"\Psi = \mu N \left[\lambda_r \beta + \ln\left(\beta / \sinh(\beta)\right)\right]",
        param_names=["mu", "N"],
        initial_guess=[0.4, 10.0],
        bounds=[(1e-6, None), (1.0, None)],  # mu > 0, N >= 1
    )
    def ArrudaBoyce(I1, params):
        """Arruda-Boyce (8-chain) model."""
        mu = params["mu"]
        N = params["N"]
        lambda_r = sp.sqrt(I1 / (3.0 * N))
        beta = inv_Langevin_Kroger(lambda_r)
        return mu * N * (lambda_r * beta + sp.log(beta / sp.sinh(beta)))

    # --- Zhan models (closed-form PK1, no symbolic energy) --------------------

    @staticmethod
    @register_model(
        model_type="custom",
        category="micromechanical",
        formula_str=r"\Psi = \mu\left[(\lambda_1+\lambda_2+\lambda_3)^2 + 2(\lambda_1^2+\lambda_2^2+\lambda_3^2)\right]",
        param_names=["mu"],
        initial_guess=[0.5],
        bounds=[(1e-6, None)],
    )
    def ZhanGaussian(lambda_1, lambda_2, lambda_3, params):
        """Zhan Gaussian model; stress comes from ``custom_pk1``."""
        raise NotImplementedError("ZhanGaussian is evaluated through custom_pk1, not a symbolic energy.")

    @staticmethod
    @register_model(
        model_type="custom",
        category="micromechanical",
        formula_str=r"\Psi = \mu\sqrt{N}\int_{\mathbb{S}^2} \left( \lambda_i \beta + \ln\left( \beta / \sinh(\beta) \right) \right) d\Omega",
        param_names=["mu", "N"],
        initial_guess=[0.5, 60.0],
        bounds=[(1e-6, None), (1.0, None)],
    )
    def ZhanNonGaussian(lambda_1, lambda_2, lambda_3, params):
        """Zhan non-Gaussian model; stress comes from ``custom_pk1``."""
        raise NotImplementedError("ZhanNonGaussian is evaluated through custom_pk1, not a symbolic energy.")

    # --- Stretch based ---------------------------------------------------------

    @staticmethod
    @register_model(
        model_type="stretch_based",
        category="phenomenological",
        formula_str=r"\Psi = \sum_i \frac{\mu_i}{\alpha_i}\left(\lambda_1^{\alpha_i} + \lambda_2^{\alpha_i} + \lambda_3^{\alpha_i} - 3\right)",
        param_names=["mu", "alpha"],
        initial_guess=[0.5, 2.0],
        bounds=[(1e-6, None), (None, None)],  # mu > 0
    )
    def Ogden(lambda_1, lambda_2, lambda_3, params):
        """Ogden model (single term)."""
        mu = params["mu"]
        alpha = params["alpha"]
        return (mu / alpha) * (lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3)

    @staticmethod
    @register_model(
        model_type="stretch_based",
        category="phenomenological",
        formula_str=r"\Psi = \frac{2\mu}{\alpha^2}\left(\lambda_1^{\alpha} + \lambda_2^{\alpha} + \lambda_3^{\alpha} - 3\right)",
        param_names=["mu", "alpha"],
        initial_guess=[0.5, 2.0],
        bounds=[(1e-6, None), (None, None)],
    )
    def ModifiedOgden(lambda_1, lambda_2, lambda_3, params):
        """Modified Ogden model (Budday et al.)."""
        mu = params["mu"]
        alpha = params["alpha"]
        return (2.0 * mu / (alpha**2)) * (lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3)

    # --- Factories --------------------------------------------------------------

    @staticmethod
    def create_ogden_model(n_terms):
        """Generate an n-term Ogden model.

        One term uses parameters ``(mu, alpha)``; multiple terms use
        ``(mu1, alpha1, mu2, alpha2, ...)`` with alternating sign conventions
        in the initial guesses to break symmetry between the terms.
        """
        n_terms = int(n_terms)
        if n_terms < 1:
            raise ValueError("Ogden model must have at least one term.")

        if n_terms == 1:
            # Match the metadata of the static one-term Ogden model: previously
            # the factory advertised a positive initial guess with negative-only
            # bounds, which made the default start point infeasible.
            param_names = ["mu", "alpha"]
            initial_guess = list(MaterialModels.Ogden.initial_guess)
            bounds = list(MaterialModels.Ogden.bounds)
        else:
            param_names = []
            initial_guess = []
            bounds = []
            for i in range(1, n_terms + 1):
                param_names.extend([f"mu{i}", f"alpha{i}"])
                if i == 1:
                    initial_guess.extend([-0.5, -2.0])
                    bounds.extend([(None, -1e-6), (None, -1e-6)])
                else:
                    initial_guess.extend([0.5 / i, 2.0 * i])
                    bounds.extend([(1e-6, None), (1e-6, None)])

        def Ogden_Dynamic(lambda_1, lambda_2, lambda_3, params):
            psi = 0
            for i in range(1, n_terms + 1):
                mu = params["mu" if n_terms == 1 else f"mu{i}"]
                alpha = params["alpha" if n_terms == 1 else f"alpha{i}"]
                psi += (mu / alpha) * (lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3)
            return psi

        Ogden_Dynamic.__name__ = f"Ogden_{n_terms}term"
        Ogden_Dynamic.model_type = "stretch_based"
        Ogden_Dynamic.category = "phenomenological"
        Ogden_Dynamic.formula = (
            rf"\Psi = \sum_{{i=1}}^{{{n_terms}}} \frac{{\mu_i}}{{\alpha_i}}"
            rf"\left(\lambda_1^{{\alpha_i}} + \lambda_2^{{\alpha_i}} + \lambda_3^{{\alpha_i}} - 3\right)"
        )
        Ogden_Dynamic.param_names = param_names
        Ogden_Dynamic.initial_guess = initial_guess
        Ogden_Dynamic.bounds = bounds
        return Ogden_Dynamic

    @staticmethod
    def create_hill_model(strain_name):
        """Generate a Hill model for a generalized strain measure.

        Parameters are ``mu`` plus the strain parameters suffixed with ``1``
        (e.g. ``m1``, ``n1``).
        """
        if strain_name not in STRAIN_CONFIGS:
            raise ValueError(f"Unknown strain configuration: {strain_name}")

        config = STRAIN_CONFIGS[strain_name]
        strain_params = config["params"]
        strain_func = strain_function(strain_name)

        full_param_names = ["mu"] + [f"{p}1" for p in strain_params]
        full_initial_guess = [10.0] + config["defaults"]
        full_bounds = [(1e-6, None)] + config["bounds"]  # mu > 0

        def Hill_Dynamic(lambda_1, lambda_2, lambda_3, params):
            mu = params["mu"]
            local = {p: params[f"{p}1"] for p in strain_params}
            E1 = strain_func(lambda_1, local)
            E2 = strain_func(lambda_2, local)
            E3 = strain_func(lambda_3, local)
            return mu * (E1**2 + E2**2 + E3**2)

        Hill_Dynamic.__name__ = f"Hill_{strain_name}"
        Hill_Dynamic.model_type = "stretch_based"
        Hill_Dynamic.category = "phenomenological"
        Hill_Dynamic.formula = r"\Psi = \mu \sum_{i=1}^3 E(\lambda_i)^2"
        Hill_Dynamic.strain_formula = STRAIN_FORMULAS.get(strain_name, "")
        Hill_Dynamic.param_names = full_param_names
        Hill_Dynamic.initial_guess = full_initial_guess
        Hill_Dynamic.bounds = full_bounds
        return Hill_Dynamic


# Closed-form PK1 evaluators for the custom models (scalar + batch).
MaterialModels.ZhanGaussian.custom_pk1 = zhan_gaussian_pk1
MaterialModels.ZhanGaussian.custom_pk1_batch = zhan_gaussian_pk1_batch
MaterialModels.ZhanNonGaussian.custom_pk1 = zhan_nongaussian_pk1
MaterialModels.ZhanNonGaussian.custom_pk1_batch = zhan_nongaussian_pk1_batch
