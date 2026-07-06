"""Stress evaluation for hyperelastic models under incompressibility.

The :class:`Kinematics` solver turns a tagged energy function (see
``models.register_model``) into PK1/PK2/Cauchy stress evaluators. Symbolic
derivatives are prepared once with sympy and lambdified to numpy.

Two evaluation paths are provided:

* scalar methods (``get_1st_PK_stress`` ...) operating on a single ``F``;
* batch methods (``first_pk_batch`` ...) operating on a stack ``(n, 3, 3)``.

The batch methods perform the same floating-point operations element-wise
and reproduce the scalar results bit-for-bit; they exist because parameter
calibration evaluates thousands of stress states per objective call.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

_EYE3 = np.eye(3)


def _as_full(value, n):
    """Broadcast a lambdify result to shape ``(n,)`` (constants collapse)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return arr


class Kinematics:
    """Stress solver for a tagged hyperelastic energy function.

    Args:
        energy_function: Function tagged by ``models.register_model`` (or a
            ``ParallelNetwork``). The ``model_type`` tag selects the pipeline:
            ``invariant_based``, ``stretch_based`` or ``custom``.
        param_names: Ordered parameter names for the optimizer vector.
        use_spectral_stretch: For stretch-based models, use a spectral
            decomposition when ``F`` has off-diagonal entries.
        spectral_tol: Off-diagonal magnitude that triggers the spectral path.
    """

    def __init__(self, energy_function, param_names, use_spectral_stretch=True, spectral_tol=1e-10):
        self.energy_function = energy_function
        self.param_names_ordered = list(param_names)
        self.use_spectral_stretch = use_spectral_stretch
        self.spectral_tol = spectral_tol

        tag_type = getattr(energy_function, "model_type", None)
        if tag_type is None:
            raise ValueError(f"Function {energy_function.__name__} missing 'model_type' tag.")
        if tag_type == "invariant_based":
            self.model_type = "invariant"
        elif tag_type == "stretch_based":
            self.model_type = "stretch"
        elif tag_type == "custom":
            self.model_type = "custom"
        else:
            raise ValueError(f"Unknown model_type tag: {tag_type}")

        if self.model_type != "custom":
            self.param_symbols = {name: sp.Symbol(name) for name in self.param_names_ordered}
            self.param_symbols_list = [self.param_symbols[name] for name in self.param_names_ordered]

        if self.model_type == "invariant":
            self._prepare_invariant_derivatives()
        elif self.model_type == "stretch":
            self._prepare_stretch_derivatives()

    # --- symbolic preparation ------------------------------------------------

    def _prepare_invariant_derivatives(self):
        """Lambdify dPsi/dI1 and dPsi/dI2."""
        I1, I2 = sp.symbols("I1 I2")

        # Energy signature is (I1, params) or (I1, I2, params).
        try:
            psi_expr = self.energy_function(I1, self.param_symbols)
        except TypeError:
            psi_expr = self.energy_function(I1, I2, self.param_symbols)

        inputs = [I1, I2] + self.param_symbols_list
        self.calc_dPsi_dI1 = sp.lambdify(inputs, sp.diff(psi_expr, I1), modules="numpy")
        self.calc_dPsi_dI2 = sp.lambdify(inputs, sp.diff(psi_expr, I2), modules="numpy")

    def _prepare_stretch_derivatives(self):
        """Lambdify dPsi/dlambda_i."""
        l1, l2, l3 = sp.symbols("lambda_1 lambda_2 lambda_3")
        psi_expr = self.energy_function(l1, l2, l3, self.param_symbols)

        inputs = [l1, l2, l3] + self.param_symbols_list
        self.calc_dPsi_dl1 = sp.lambdify(inputs, sp.diff(psi_expr, l1), modules="numpy")
        self.calc_dPsi_dl2 = sp.lambdify(inputs, sp.diff(psi_expr, l2), modules="numpy")
        self.calc_dPsi_dl3 = sp.lambdify(inputs, sp.diff(psi_expr, l3), modules="numpy")

    def _param_values(self, params):
        return [params[name] for name in self.param_names_ordered]

    # --- scalar evaluation ---------------------------------------------------

    def _should_use_spectral_stretch(self, F):
        if not self.use_spectral_stretch:
            return False
        off_diag = F - np.diag(np.diag(F))
        return np.any(np.abs(off_diag) > self.spectral_tol)

    def get_2nd_PK_stress(self, F, params):
        """Total second Piola-Kirchhoff stress with the incompressibility
        pressure eliminated through the plane-stress condition sigma_33 = 0."""
        p_vals = self._param_values(params)

        if self.model_type == "invariant":
            C = F.T @ F
            I1 = np.trace(C)
            C2 = C @ C
            I2 = 0.5 * (I1**2 - np.trace(C2))

            psi1 = self.calc_dPsi_dI1(I1, I2, *p_vals)
            psi2 = self.calc_dPsi_dI2(I1, I2, *p_vals)

            S_hyper = 2.0 * ((psi1 + I1 * psi2) * _EYE3 - psi2 * C)

        elif self.model_type == "stretch":
            if self._should_use_spectral_stretch(F):
                C = F.T @ F
                eigvals, eigvecs = np.linalg.eigh(C)
                lambdas = np.sqrt(np.clip(eigvals, 1e-12, None))

                l1, l2, l3 = lambdas
                s1 = self.calc_dPsi_dl1(l1, l2, l3, *p_vals)
                s2 = self.calc_dPsi_dl2(l1, l2, l3, *p_vals)
                s3 = self.calc_dPsi_dl3(l1, l2, l3, *p_vals)

                S_diag = np.diag([s1 / l1, s2 / l2, s3 / l3])
                S_hyper = eigvecs @ S_diag @ eigvecs.T
            else:
                l1, l2, l3 = F[0, 0], F[1, 1], F[2, 2]

                s1 = self.calc_dPsi_dl1(l1, l2, l3, *p_vals)
                s2 = self.calc_dPsi_dl2(l1, l2, l3, *p_vals)
                s3 = self.calc_dPsi_dl3(l1, l2, l3, *p_vals)

                S_hyper = np.diag([s1 / l1, s2 / l2, s3 / l3])
        else:
            P = self.energy_function.custom_pk1(F, params)
            return np.linalg.solve(F, P)

        Sigma_hyper = F @ S_hyper @ F.T
        p = Sigma_hyper[2, 2]

        C = F.T @ F
        C_inv = np.linalg.inv(C)
        return S_hyper - p * C_inv

    def get_1st_PK_stress(self, F, params):
        if self.model_type == "custom":
            return self.energy_function.custom_pk1(F, params)
        return F @ self.get_2nd_PK_stress(F, params)

    def get_Cauchy_stress(self, F, params):
        if self.model_type == "custom":
            P = self.energy_function.custom_pk1(F, params)
            J = np.linalg.det(F)
            return (P @ F.T) / J
        S = self.get_2nd_PK_stress(F, params)
        return F @ S @ F.T

    # --- batch evaluation ----------------------------------------------------

    def _custom_pk1_batch(self, F, params):
        batch_fn = getattr(self.energy_function, "custom_pk1_batch", None)
        if batch_fn is not None:
            return batch_fn(F, params)
        return np.stack([self.energy_function.custom_pk1(Fi, params) for Fi in F])

    def _hyper_second_pk_batch(self, F, params):
        """Purely hyperelastic PK2 stack, before pressure elimination."""
        n = F.shape[0]
        p_vals = self._param_values(params)

        if self.model_type == "invariant":
            C = np.matmul(F.transpose(0, 2, 1), F)
            I1 = np.trace(C, axis1=1, axis2=2)
            C2 = np.matmul(C, C)
            I2 = 0.5 * (I1**2 - np.trace(C2, axis1=1, axis2=2))

            psi1 = _as_full(self.calc_dPsi_dI1(I1, I2, *p_vals), n)
            psi2 = _as_full(self.calc_dPsi_dI2(I1, I2, *p_vals), n)

            return 2.0 * ((psi1 + I1 * psi2)[:, None, None] * _EYE3 - psi2[:, None, None] * C)

        # stretch-based
        lams = np.empty((n, 3))
        idx = np.arange(3)
        diag_part = np.zeros_like(F)
        diag_part[:, idx, idx] = F[:, idx, idx]
        spectral = np.zeros(n, dtype=bool)
        if self.use_spectral_stretch:
            spectral = np.any(np.abs(F - diag_part) > self.spectral_tol, axis=(1, 2))

        vecs_spec = None
        if spectral.any():
            C_spec = np.matmul(F[spectral].transpose(0, 2, 1), F[spectral])
            eigvals, vecs_spec = np.linalg.eigh(C_spec)
            lams[spectral] = np.sqrt(np.clip(eigvals, 1e-12, None))
        if (~spectral).any():
            lams[~spectral] = F[~spectral][:, idx, idx]

        l1, l2, l3 = lams[:, 0], lams[:, 1], lams[:, 2]
        s1 = _as_full(self.calc_dPsi_dl1(l1, l2, l3, *p_vals), n)
        s2 = _as_full(self.calc_dPsi_dl2(l1, l2, l3, *p_vals), n)
        s3 = _as_full(self.calc_dPsi_dl3(l1, l2, l3, *p_vals), n)

        entries = np.stack([s1 / l1, s2 / l2, s3 / l3], axis=1)
        S_hyper = np.zeros((n, 3, 3))
        S_hyper[:, idx, idx] = entries
        if spectral.any():
            S_diag = S_hyper[spectral]
            S_hyper[spectral] = np.matmul(np.matmul(vecs_spec, S_diag), vecs_spec.transpose(0, 2, 1))
        return S_hyper

    def second_pk_batch(self, F, params):
        """Batched :meth:`get_2nd_PK_stress` for a stack ``(n, 3, 3)``."""
        F = np.asarray(F, dtype=float)
        if self.model_type == "custom":
            return np.linalg.solve(F, self._custom_pk1_batch(F, params))

        S_hyper = self._hyper_second_pk_batch(F, params)
        Sigma_hyper = np.matmul(np.matmul(F, S_hyper), F.transpose(0, 2, 1))
        p = Sigma_hyper[:, 2, 2]

        C = np.matmul(F.transpose(0, 2, 1), F)
        C_inv = np.linalg.inv(C)
        return S_hyper - p[:, None, None] * C_inv

    def first_pk_batch(self, F, params):
        """Batched :meth:`get_1st_PK_stress` for a stack ``(n, 3, 3)``."""
        F = np.asarray(F, dtype=float)
        if self.model_type == "custom":
            return self._custom_pk1_batch(F, params)
        return np.matmul(F, self.second_pk_batch(F, params))

    def cauchy_batch(self, F, params):
        """Batched :meth:`get_Cauchy_stress` for a stack ``(n, 3, 3)``."""
        F = np.asarray(F, dtype=float)
        if self.model_type == "custom":
            P = self._custom_pk1_batch(F, params)
            J = np.linalg.det(F)
            return np.matmul(P, F.transpose(0, 2, 1)) / J[:, None, None]
        S = self.second_pk_batch(F, params)
        return np.matmul(np.matmul(F, S), F.transpose(0, 2, 1))
