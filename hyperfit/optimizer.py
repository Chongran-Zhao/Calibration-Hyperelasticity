"""Parameter calibration for hyperelastic models.

The objective is the sum over datasets of ``SS_res / SS_tot`` (a "1 - R^2"
style normalization), so datasets with different stress magnitudes and modes
contribute comparably.
"""

from __future__ import annotations

import inspect
import logging
import threading

import numpy as np
from scipy.optimize import OptimizeResult, least_squares, minimize

from .evaluation import experimental_values, predict_observables

logger = logging.getLogger(__name__)

DEFAULT_ABS_TOL = 0.1
DEFAULT_REL_TOL = 1e-3
DEFAULT_MAX_ITER = 2000
DEFAULT_R2_TARGET = 0.95
DEFAULT_MAX_LOSS = 1.0

# User-facing method names -> (scipy backend, solver method)
METHOD_ALIASES = {
    "L-BFGS-B": ("minimize", "L-BFGS-B"),
    "trust-constr": ("minimize", "trust-constr"),
    "CG": ("minimize", "CG"),
    "Newton-CG": ("minimize", "Newton-CG"),
    "trf": ("least_squares", "trf"),
    "dogbox": ("least_squares", "dogbox"),
    "lm": ("least_squares", "lm"),
    "Trust-Region Reflective (lsqnonlin)": ("least_squares", "trf"),
    "Dogbox (lsqnonlin)": ("least_squares", "dogbox"),
    "Levenberg-Marquardt (lsqnonlin)": ("least_squares", "lm"),
}


class OptimizationAbortedError(RuntimeError):
    """Raised when optimization is aborted by user request."""


class _NullProgressBar:
    """No-op stand-in for tqdm when progress display is disabled."""

    def update(self, n=1):
        pass

    def set_postfix(self, **kwargs):
        pass

    def close(self):
        pass


def _make_progress_bar(enabled):
    if not enabled:
        return _NullProgressBar()
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullProgressBar()
    return tqdm(
        desc="  Fitting",
        unit="iter",
        bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}{postfix}]",
    )


class MaterialOptimizer:
    """Calibrates model parameters against one or more datasets.

    Args:
        kinematics_solver: A :class:`hyperfit.kinematics.Kinematics` instance.
        experimental_data: List of dataset dicts from :mod:`hyperfit.datasets`.
    """

    def __init__(self, kinematics_solver, experimental_data):
        self.solver = kinematics_solver
        self.data = experimental_data
        self.param_names = kinematics_solver.param_names_ordered
        self._current_loss = float("inf")
        self._stop_event = threading.Event()
        self._last_params = np.array([], dtype=float)

        # Per-dataset observed series and total sum of squares.
        self._exp_values = [experimental_values(ds) for ds in self.data]
        self.ss_tot_list = []
        for exp in self._exp_values:
            stress_flat = np.asarray(exp).ravel()
            ss_tot = np.sum((stress_flat - np.mean(stress_flat)) ** 2)
            if ss_tot < 1e-12:
                ss_tot = 1.0
            self.ss_tot_list.append(ss_tot)

    # --- cooperative cancellation --------------------------------------------

    def request_stop(self):
        self._stop_event.set()

    def clear_stop(self):
        self._stop_event.clear()

    def _check_stop(self):
        if self._stop_event.is_set():
            raise OptimizationAbortedError("Optimization aborted by user.")

    # --- objective pieces ------------------------------------------------------

    def _dataset_diff(self, index, params_dict):
        """Model-minus-experiment for one dataset, in native shape."""
        model = predict_observables(self.solver, self.data[index], params_dict)
        return model - self._exp_values[index]

    def _objective_function(self, params_array):
        """Sum over datasets of SS_res / SS_tot."""
        self._check_stop()
        self._last_params = np.array(params_array, dtype=float, copy=True)
        params_dict = dict(zip(self.param_names, params_array))

        total_normalized_error = 0.0
        for i in range(len(self.data)):
            diff = self._dataset_diff(i, params_dict)
            total_normalized_error += np.sum(diff**2) / self.ss_tot_list[i]

        self._current_loss = total_normalized_error
        return total_normalized_error

    def _residuals(self, params_array):
        """Residual vector such that sum(residuals^2) equals the objective."""
        self._check_stop()
        self._last_params = np.array(params_array, dtype=float, copy=True)
        params_dict = dict(zip(self.param_names, params_array))

        residuals = []
        for i in range(len(self.data)):
            diff = np.asarray(self._dataset_diff(i, params_dict), dtype=float)
            if diff.ndim > 1:
                diff = diff.reshape(-1)
            ss_tot = self.ss_tot_list[i]
            norm = np.sqrt(ss_tot) if ss_tot > 0 else 1.0
            residuals.append(diff / norm)

        residuals = np.concatenate(residuals) if residuals else np.array([], dtype=float)
        self._current_loss = float(np.sum(residuals**2))
        return residuals

    def compute_r2(self, params_array):
        """Pooled and per-dataset-averaged coefficients of determination."""
        params_dict = dict(zip(self.param_names, params_array))
        ss_res_total = 0.0
        ss_tot_total = 0.0
        r2_values = []

        for i in range(len(self.data)):
            diff = self._dataset_diff(i, params_dict)
            ss_res = np.sum(diff**2)
            ss_tot = self.ss_tot_list[i]
            ss_res_total += ss_res
            ss_tot_total += ss_tot
            r2_values.append(1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0)

        r2_total = 1.0 - (ss_res_total / ss_tot_total) if ss_tot_total > 1e-12 else 0.0
        r2_avg = float(np.mean(r2_values)) if r2_values else 0.0
        return r2_total, r2_avg

    # --- fitting -----------------------------------------------------------------

    def _numeric_grad(self, xk):
        """Central-difference gradient (only used by Newton-CG)."""
        eps = 1e-6
        grad = np.zeros_like(xk, dtype=float)
        for j in range(len(xk)):
            x_fwd = np.array(xk, dtype=float)
            x_bwd = np.array(xk, dtype=float)
            x_fwd[j] += eps
            x_bwd[j] -= eps
            grad[j] = (self._objective_function(x_fwd) - self._objective_function(x_bwd)) / (2.0 * eps)
        return grad

    def fit(
        self,
        initial_guess,
        bounds=None,
        method="L-BFGS-B",
        progress_cb=None,
        abs_tol=DEFAULT_ABS_TOL,
        rel_tol=DEFAULT_REL_TOL,
        max_iter=DEFAULT_MAX_ITER,
        r2_target=DEFAULT_R2_TARGET,
        max_loss=DEFAULT_MAX_LOSS,
        show_progress=False,
    ):
        """Run the calibration.

        Args:
            initial_guess: Starting parameter vector.
            bounds: Optional ``[(min, max), ...]`` per parameter.
            method: One of the keys in :data:`METHOD_ALIASES`.
            progress_cb: Optional callback ``(iteration, params, loss)``.
            abs_tol / rel_tol / max_iter / r2_target / max_loss: Convergence
                and acceptance criteria evaluated after the solver finishes.
            show_progress: Display a tqdm progress bar (CLI use).

        Returns:
            ``scipy.optimize.OptimizeResult`` enriched with ``r2_total``,
            ``r2_avg``, ``converged``, ``aborted``, ``initial_loss`` and
            ``rel_improvement``.
        """
        method_key = (method or "L-BFGS-B").strip()
        method_entry = (
            METHOD_ALIASES.get(method_key)
            or METHOD_ALIASES.get(method_key.lower())
            or METHOD_ALIASES.get(method_key.upper())
        )
        if method_entry is None:
            raise ValueError(f"Unsupported method: {method}")
        backend, solver_method = method_entry

        logger.info("Starting optimization using %s", method_key)
        self.clear_stop()
        self._last_params = np.array(initial_guess, dtype=float, copy=True)
        initial_loss = float(self._objective_function(initial_guess))

        pbar = _make_progress_bar(show_progress)
        iter_count = 0

        def callback_minimize(xk, *args):
            nonlocal iter_count
            self._check_stop()
            self._last_params = np.array(xk, dtype=float, copy=True)
            iter_count += 1
            pbar.set_postfix(Loss=f"{self._current_loss:.6f}")
            pbar.update(1)
            if progress_cb is not None:
                progress_cb(iter_count, np.array(xk, dtype=float), self._current_loss)

        had_exception = False
        was_aborted = False
        result = None
        try:
            if backend == "minimize":
                use_bounds = bounds if solver_method in {"L-BFGS-B", "trust-constr"} else None
                jac = self._numeric_grad if solver_method == "Newton-CG" else None

                result = minimize(
                    fun=self._objective_function,
                    x0=initial_guess,
                    method=solver_method,
                    bounds=use_bounds,
                    jac=jac,
                    callback=callback_minimize,
                    options={"maxiter": max_iter, "disp": False},
                )
            else:  # least_squares
                if bounds is None:
                    lower = np.full(len(initial_guess), -np.inf)
                    upper = np.full(len(initial_guess), np.inf)
                else:
                    lower = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds], dtype=float)
                    upper = np.array([(np.inf if b[1] is None else b[1]) for b in bounds], dtype=float)
                if solver_method == "lm":  # lm does not support bounds
                    lower = np.full(len(initial_guess), -np.inf)
                    upper = np.full(len(initial_guess), np.inf)

                use_callback = "callback" in inspect.signature(least_squares).parameters
                eval_count = 0
                report_every = 5

                def residuals_with_progress(xk):
                    nonlocal eval_count, iter_count
                    self._last_params = np.array(xk, dtype=float, copy=True)
                    res = self._residuals(xk)
                    eval_count += 1
                    if progress_cb is not None and not use_callback:
                        if eval_count == 1 or eval_count % report_every == 0:
                            iter_count += 1
                            pbar.set_postfix(Loss=f"{self._current_loss:.6f}")
                            pbar.update(1)
                            progress_cb(iter_count, np.array(xk, dtype=float), self._current_loss)
                    return res

                lsq_kwargs = {
                    "fun": residuals_with_progress,
                    "x0": initial_guess,
                    "method": solver_method,
                    "bounds": (lower, upper),
                    "max_nfev": max_iter,
                }
                if use_callback:
                    def lsq_callback(xk, *args, **kwargs):
                        nonlocal iter_count
                        self._check_stop()
                        self._last_params = np.array(xk, dtype=float, copy=True)
                        iter_count += 1
                        loss = float(self._objective_function(xk))
                        pbar.set_postfix(Loss=f"{loss:.6f}")
                        pbar.update(1)
                        if progress_cb is not None:
                            progress_cb(iter_count, np.array(xk, dtype=float), loss)

                    lsq_kwargs["callback"] = lsq_callback

                result = least_squares(**lsq_kwargs)
                result.fun = float(np.sum(result.fun**2))
                if not hasattr(result, "nit"):
                    result.nit = iter_count if iter_count else getattr(result, "nfev", 0)

        except OptimizationAbortedError as exc:
            logger.info("Optimization aborted: %s", exc)
            x_abort = self._last_params if self._last_params.size else np.array(initial_guess, dtype=float)
            result = OptimizeResult(success=False, message=str(exc), fun=self._current_loss, x=x_abort)
            had_exception = True
            was_aborted = True
        except Exception as exc:
            logger.warning("Optimization error: %s", exc)
            result = OptimizeResult(success=False, message=str(exc), fun=self._current_loss, x=initial_guess)
            had_exception = True
        finally:
            pbar.close()

        # Acceptance: absolute loss, relative improvement, R^2 target, max loss.
        final_loss = float(getattr(result, "fun", self._current_loss))
        iter_used = int(getattr(result, "nit", iter_count) or iter_count)
        rel_improvement = 0.0
        if initial_loss > 0:
            rel_improvement = abs(initial_loss - final_loss) / max(initial_loss, 1e-12)

        if was_aborted:
            r2_total, r2_avg = 0.0, 0.0
        else:
            r2_total, r2_avg = self.compute_r2(getattr(result, "x", initial_guess))
        result.r2_total = r2_total
        result.r2_avg = r2_avg
        result.r2_target = r2_target
        result.aborted = was_aborted

        criteria_met = []
        max_loss_exceeded = max_loss is not None and final_loss > max_loss
        if r2_target is not None and r2_total >= r2_target:
            criteria_met.append("r2")
        if abs_tol is not None and final_loss <= abs_tol:
            criteria_met.append("abs_tol")
        if rel_tol is not None and rel_improvement >= rel_tol:
            criteria_met.append("rel_tol")

        converged = bool(criteria_met) and not max_loss_exceeded
        if had_exception:
            converged = False
        if was_aborted:
            result.success = False
            result.message = "Optimization aborted by user."
        elif converged:
            result.success = True
        else:
            result.success = False
            reasons = []
            if max_loss_exceeded:
                reasons.append("max_loss")
            if r2_target is not None and r2_total < r2_target:
                reasons.append("r2")
            if abs_tol is not None and final_loss > abs_tol:
                reasons.append("abs_tol")
            if rel_tol is not None and rel_improvement < rel_tol:
                reasons.append("rel_tol")
            if iter_used >= max_iter:
                reasons.append("max_iter")
            if not reasons:
                reasons.append("criteria")
            result.message = f"Convergence criteria not met: {', '.join(reasons)}"

        result.converged = converged
        result.initial_loss = initial_loss
        result.rel_improvement = rel_improvement
        result.nit = iter_used
        return result
