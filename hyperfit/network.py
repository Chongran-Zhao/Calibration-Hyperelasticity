"""Composite model: parallel network of hyperelastic branches.

Physics: isostrain composition, ``F_total = F_i`` for every branch and
``Psi_total = sum_i Psi_i``. Branch parameters are namespaced with the branch
prefix (``C1`` of branch ``matrix`` becomes ``matrix_C1``).
"""

from __future__ import annotations

import inspect
import logging

import numpy as np

logger = logging.getLogger(__name__)


class ParallelNetwork:
    """Container aggregating several material models in parallel.

    The instance masquerades as a tagged energy function, so it can be handed
    to :class:`hyperfit.kinematics.Kinematics` directly. If any branch is a
    custom-stress model the whole network switches to the custom-PK1 path.
    """

    def __init__(self):
        self.components = []

        # Flat parameter metadata for the optimizer.
        self.param_names = []
        self.initial_guess = []
        self.bounds = []

        # Tags mirroring register_model so Kinematics can consume the network.
        self.model_type = "stretch_based"
        self.category = "composite"
        self.formula = r"\Psi_{total} = \sum_{i=1}^{N} \Psi_{i}"

    def add_model(self, model_func, name_prefix):
        """Add a branch.

        Args:
            model_func: Tagged model function from :class:`MaterialModels`.
            name_prefix: Unique branch identifier used to namespace parameters.
        """
        orig_params = getattr(model_func, "param_names", [])
        orig_guess = getattr(model_func, "initial_guess", [])
        orig_bounds = getattr(model_func, "bounds", [])

        new_params = [f"{name_prefix}_{p}" for p in orig_params]

        self.components.append({
            "func": model_func,
            "prefix": name_prefix,
            "local_params": orig_params,
            "global_params": new_params,
            "orig_type": getattr(model_func, "model_type", "stretch_based"),
            "solver": None,
        })

        self.param_names.extend(new_params)

        # Perturb the initial guess of every branch after the first, so
        # identical branches do not start on a symmetric optimization path.
        # Branch 2: +10%, branch 3: -10%, branch 4: +20%, ... deterministic.
        branch_idx = len(self.components)
        perturbed_guess = []
        for val in orig_guess:
            if branch_idx == 1:
                factor = 1.0
            else:
                sign = 1.0 if branch_idx % 2 == 0 else -1.0
                factor = 1.0 + sign * 0.1 * (branch_idx // 2 + 1)
            new_val = val * factor
            if val > 0 and new_val <= 0:  # never flip the sign of a modulus
                new_val = val * 0.5
            perturbed_guess.append(new_val)

        logger.debug(
            "ParallelNetwork: added branch '%s' (%s), initial guess %s -> %s",
            name_prefix, model_func.__name__, orig_guess, perturbed_guess,
        )
        self.initial_guess.extend(perturbed_guess)

        if orig_bounds:
            self.bounds.extend(orig_bounds)
        else:
            self.bounds.extend([(None, None)] * len(new_params))

        if getattr(model_func, "model_type", None) == "custom":
            self.model_type = "custom"

    def _local_params(self, comp, params):
        return {lp: params[gp] for lp, gp in zip(comp["local_params"], comp["global_params"])}

    def __call__(self, lambda_1, lambda_2, lambda_3, params):
        """Total strain energy density (symbolic path).

        Invariant-based branches are fed I1/I2 expressed through the principal
        stretches under incompressibility (J = 1).
        """
        total_psi = 0
        I1 = lambda_1**2 + lambda_2**2 + lambda_3**2
        I2 = lambda_1**(-2) + lambda_2**(-2) + lambda_3**(-2)

        for comp in self.components:
            model_func = comp["func"]
            local = self._local_params(comp, params)
            if comp["orig_type"] == "invariant_based":
                if "I2" in inspect.signature(model_func).parameters:
                    total_psi += model_func(I1, I2, local)
                else:
                    total_psi += model_func(I1, local)
            else:
                total_psi += model_func(lambda_1, lambda_2, lambda_3, local)
        return total_psi

    def _component_solver(self, comp):
        if comp["solver"] is None:
            from .kinematics import Kinematics

            comp["solver"] = Kinematics(comp["func"], comp["local_params"])
        return comp["solver"]

    def custom_pk1(self, F, params):
        """Total PK1 stress for networks containing custom-stress branches."""
        total = np.zeros((3, 3), dtype=float)
        for comp in self.components:
            local = self._local_params(comp, params)
            if comp["orig_type"] == "custom":
                total += comp["func"].custom_pk1(F, local)
            else:
                total += self._component_solver(comp).get_1st_PK_stress(F, local)
        return total

    def custom_pk1_batch(self, F, params):
        """Batched :meth:`custom_pk1` for a stack ``(n, 3, 3)``."""
        total = np.zeros(F.shape, dtype=float)
        for comp in self.components:
            local = self._local_params(comp, params)
            if comp["orig_type"] == "custom":
                batch_fn = getattr(comp["func"], "custom_pk1_batch", None)
                if batch_fn is not None:
                    total += batch_fn(F, local)
                else:
                    total += np.stack([comp["func"].custom_pk1(Fi, local) for Fi in F])
            else:
                total += self._component_solver(comp).first_pk_batch(F, local)
        return total
