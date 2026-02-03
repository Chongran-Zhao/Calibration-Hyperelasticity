import inspect
import numpy as np
from scipy.optimize import minimize, least_squares
from utils import get_stress_components
from tqdm import tqdm

DEFAULT_ABS_TOL = 0.1
DEFAULT_REL_TOL = 1e-3
DEFAULT_MAX_ITER = 2000
DEFAULT_R2_TARGET = 0.95
DEFAULT_MAX_LOSS = 1.0

class MaterialOptimizer:
    """
    Optimizer for calibrating hyperelastic material parameters.
    Uses R^2 (Coefficient of Determination) based objective function to normalize
    errors across different datasets (UT, ET, PS).
    """

    def __init__(self, kinematics_solver, experimental_data):
        self.solver = kinematics_solver
        self.data = experimental_data
        self.param_names = kinematics_solver.param_names_ordered
        self._current_loss = float('inf')
        
        # Pre-calculate SS_tot (Total Sum of Squares) for each dataset
        self.ss_tot_list = []
        for dataset in self.data:
            stress_exp = dataset['stress_exp']
            bt_diff = dataset.get('bt_component') == 'diff' and dataset.get('mode') == 'BT'
            component = dataset.get('component')
            if bt_diff and np.ndim(stress_exp) > 1:
                stress_flat = np.array(stress_exp[:, 0] - stress_exp[:, 1]).ravel()
            elif component == "22":
                stress_flat = np.array(stress_exp).ravel()
            else:
                stress_flat = np.array(stress_exp).ravel()
            mean_stress = np.mean(stress_flat)
            ss_tot = np.sum((stress_flat - mean_stress)**2)
            
            if ss_tot < 1e-12:
                ss_tot = 1.0 
            
            self.ss_tot_list.append(ss_tot)

    def _objective_function(self, params_array):
        """
        Loss function to minimize.
        Minimizes Sum of (1 - R^2) for all datasets.
        """
        params_dict = dict(zip(self.param_names, params_array))
        total_normalized_error = 0.0
        
        for i, dataset in enumerate(self.data):
            mode = dataset['mode']
            f_list = dataset['F_list']
            stress_type = dataset.get('stress_type', 'PK1')
            stress_exp = dataset['stress_exp']
            ss_tot = self.ss_tot_list[i]
            bt_diff = dataset.get('bt_component') == 'diff' and mode == 'BT'
            component = dataset.get('component')
            
            model_stresses = []
            for F in f_list:
                if stress_type == 'cauchy':
                    stress_tensor = self.solver.get_Cauchy_stress(F, params_dict)
                else:
                    stress_tensor = self.solver.get_1st_PK_stress(F, params_dict)
                components = get_stress_components(stress_tensor, mode)
                if bt_diff:
                    model_stresses.append(components[0] - components[1])
                elif component == "22":
                    model_stresses.append(components[1])
                elif component == "11":
                    model_stresses.append(components[0])
                elif np.ndim(stress_exp) == 1:
                    model_stresses.append(components[0])
                else:
                    model_stresses.append(components[:stress_exp.shape[1]])
            
            model_stresses = np.array(model_stresses)
            
            # Simple Sum of Squared Residuals
            if bt_diff and np.ndim(stress_exp) > 1:
                exp_values = stress_exp[:, 0] - stress_exp[:, 1]
            else:
                exp_values = stress_exp
            diff = model_stresses - exp_values
            ss_res = np.sum(diff**2)
            
            # Normalize by Variance (1 - R^2 style)
            total_normalized_error += ss_res / ss_tot
            
        self._current_loss = total_normalized_error
        return total_normalized_error

    def _residuals(self, params_array):
        """
        Residual vector for least_squares-based solvers.
        Matches the objective function: sum(residuals^2) == objective.
        """
        params_dict = dict(zip(self.param_names, params_array))
        residuals = []

        for i, dataset in enumerate(self.data):
            mode = dataset['mode']
            f_list = dataset['F_list']
            stress_type = dataset.get('stress_type', 'PK1')
            stress_exp = dataset['stress_exp']
            ss_tot = self.ss_tot_list[i]
            bt_diff = dataset.get('bt_component') == 'diff' and mode == 'BT'
            component = dataset.get('component')

            model_stresses = []
            for F in f_list:
                if stress_type == 'cauchy':
                    stress_tensor = self.solver.get_Cauchy_stress(F, params_dict)
                else:
                    stress_tensor = self.solver.get_1st_PK_stress(F, params_dict)
                components = get_stress_components(stress_tensor, mode)
                if bt_diff:
                    model_stresses.append(components[0] - components[1])
                elif component == "22":
                    model_stresses.append(components[1])
                elif component == "11":
                    model_stresses.append(components[0])
                elif np.ndim(stress_exp) == 1:
                    model_stresses.append(components[0])
                else:
                    model_stresses.append(components[:stress_exp.shape[1]])

            model_stresses = np.array(model_stresses)
            if bt_diff and np.ndim(stress_exp) > 1:
                exp_values = stress_exp[:, 0] - stress_exp[:, 1]
            else:
                exp_values = stress_exp
            diff = model_stresses - exp_values
            diff = np.array(diff, dtype=float)
            if diff.ndim > 1:
                diff = diff.reshape(-1)
            norm = np.sqrt(ss_tot) if ss_tot > 0 else 1.0
            residuals.append(diff / norm)

        residuals = np.concatenate(residuals) if residuals else np.array([], dtype=float)
        self._current_loss = float(np.sum(residuals**2))
        return residuals

    def compute_r2(self, params_array):
        params_dict = dict(zip(self.param_names, params_array))
        ss_res_total = 0.0
        ss_tot_total = 0.0
        r2_values = []

        for i, dataset in enumerate(self.data):
            mode = dataset['mode']
            f_list = dataset['F_list']
            stress_type = dataset.get('stress_type', 'PK1')
            stress_exp = dataset['stress_exp']
            ss_tot = self.ss_tot_list[i]
            bt_diff = dataset.get('bt_component') == 'diff' and mode == 'BT'
            component = dataset.get('component')

            model_stresses = []
            for F in f_list:
                if stress_type == 'cauchy':
                    stress_tensor = self.solver.get_Cauchy_stress(F, params_dict)
                else:
                    stress_tensor = self.solver.get_1st_PK_stress(F, params_dict)
                components = get_stress_components(stress_tensor, mode)
                if bt_diff:
                    model_stresses.append(components[0] - components[1])
                elif component == "22":
                    model_stresses.append(components[1])
                elif component == "11":
                    model_stresses.append(components[0])
                elif np.ndim(stress_exp) == 1:
                    model_stresses.append(components[0])
                else:
                    model_stresses.append(components[:stress_exp.shape[1]])

            model_stresses = np.array(model_stresses)
            if bt_diff and np.ndim(stress_exp) > 1:
                exp_values = stress_exp[:, 0] - stress_exp[:, 1]
            else:
                exp_values = stress_exp
            diff = model_stresses - exp_values
            ss_res = np.sum(diff**2)
            ss_res_total += ss_res
            ss_tot_total += ss_tot

            if ss_tot > 1e-12:
                r2_values.append(1.0 - (ss_res / ss_tot))
            else:
                r2_values.append(0.0)

        if ss_tot_total > 1e-12:
            r2_total = 1.0 - (ss_res_total / ss_tot_total)
        else:
            r2_total = 0.0
        r2_avg = float(np.mean(r2_values)) if r2_values else 0.0
        return r2_total, r2_avg

    def fit(
        self,
        initial_guess,
        bounds=None,
        method='L-BFGS-B',
        progress_cb=None,
        abs_tol=DEFAULT_ABS_TOL,
        rel_tol=DEFAULT_REL_TOL,
        max_iter=DEFAULT_MAX_ITER,
        r2_target=DEFAULT_R2_TARGET,
        max_loss=DEFAULT_MAX_LOSS,
    ):
        """
        Perform the optimization using the selected method.
        
        Args:
            initial_guess (list): Starting parameters.
            bounds (list): Parameter bounds [(min, max), ...].
            method (str): Optimization algorithm.
        """
        print(f"Starting optimization using {method}...")
        initial_loss = float(self._objective_function(initial_guess))
        
        # --- Progress Bar Setup ---
        # Different methods iterate differently, so we use a generic counter
        pbar = tqdm(desc=f"  Fitting", unit="iter", 
                    bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}{postfix}]")
        
        iter_count = 0

        # Callback wrapper for local minimizers (minimize)
        def callback_minimize(xk, *args):
            nonlocal iter_count
            iter_count += 1
            pbar.set_postfix(Loss=f"{self._current_loss:.6f}")
            pbar.update(1)
            if progress_cb is not None:
                progress_cb(iter_count, np.array(xk, dtype=float), self._current_loss)

        # Callback wrapper for differential_evolution
        def callback_de(xk, convergence):
            nonlocal iter_count
            iter_count += 1
            pbar.set_postfix(Loss=f"{self._current_loss:.6f}", Conv=f"{convergence:.4f}")
            pbar.update(1)
            if progress_cb is not None:
                progress_cb(iter_count, np.array(xk, dtype=float), self._current_loss)

        result = None
        method_aliases = {
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

        method_key = (method or "L-BFGS-B").strip()
        method_entry = method_aliases.get(method_key)
        if method_entry is None:
            method_entry = method_aliases.get(method_key.lower())
        if method_entry is None:
            method_entry = method_aliases.get(method_key.upper())
        if method_entry is None:
            raise ValueError(f"Unsupported method: {method}")

        backend, solver_method = method_entry

        def numeric_grad(xk):
            eps = 1e-6
            grad = np.zeros_like(xk, dtype=float)
            for j in range(len(xk)):
                x_fwd = np.array(xk, dtype=float)
                x_bwd = np.array(xk, dtype=float)
                x_fwd[j] += eps
                x_bwd[j] -= eps
                grad[j] = (self._objective_function(x_fwd) - self._objective_function(x_bwd)) / (2.0 * eps)
            return grad

        had_exception = False
        try:
            if backend == "minimize":
                use_bounds = bounds if solver_method in {'L-BFGS-B', 'trust-constr'} else None
                jac = numeric_grad if solver_method == 'Newton-CG' else None

                result = minimize(
                    fun=self._objective_function,
                    x0=initial_guess,
                    method=solver_method,
                    bounds=use_bounds,
                    jac=jac,
                    callback=callback_minimize,
                    options={'maxiter': max_iter, 'disp': False}
                )
            elif backend == "least_squares":
                if bounds is None:
                    lower = np.full(len(initial_guess), -np.inf)
                    upper = np.full(len(initial_guess), np.inf)
                else:
                    lower = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds], dtype=float)
                    upper = np.array([(np.inf if b[1] is None else b[1]) for b in bounds], dtype=float)
                if solver_method == "lm":
                    lower = np.full(len(initial_guess), -np.inf)
                    upper = np.full(len(initial_guess), np.inf)

                use_callback = "callback" in inspect.signature(least_squares).parameters
                eval_count = 0
                report_every = 5

                def residuals_with_progress(xk):
                    nonlocal eval_count, iter_count
                    res = self._residuals(xk)
                    eval_count += 1
                    if progress_cb is not None and not use_callback:
                        if eval_count == 1 or eval_count % report_every == 0:
                            iter_count += 1
                            pbar.set_postfix(Loss=f"{self._current_loss:.6f}")
                            pbar.update(1)
                            progress_cb(iter_count, np.array(xk, dtype=float), self._current_loss)
                    return res

                lsq_callback = None
                if use_callback:
                    def lsq_callback(xk, *args, **kwargs):
                        nonlocal iter_count
                        iter_count += 1
                        loss = float(self._objective_function(xk))
                        pbar.set_postfix(Loss=f"{loss:.6f}")
                        pbar.update(1)
                        if progress_cb is not None:
                            progress_cb(iter_count, np.array(xk, dtype=float), loss)

                lsq_kwargs = {
                    "fun": residuals_with_progress,
                    "x0": initial_guess,
                    "method": solver_method,
                    "bounds": (lower, upper),
                    "max_nfev": max_iter,
                }
                if use_callback:
                    lsq_kwargs["callback"] = lsq_callback
                result = least_squares(**lsq_kwargs)
                final_loss = float(np.sum(result.fun**2))
                result.fun = final_loss
                if not hasattr(result, "nit"):
                    result.nit = iter_count if iter_count else getattr(result, "nfev", 0)
            else:
                raise ValueError(f"Unsupported backend: {backend}")

        except Exception as e:
            print(f"\nOptimization Error: {e}")
            from scipy.optimize import OptimizeResult
            result = OptimizeResult(success=False, message=str(e), fun=self._current_loss, x=initial_guess)
            had_exception = True

        finally:
            pbar.close()

        # Convergence checks: absolute error, relative improvement, max iterations.
        final_loss = float(getattr(result, "fun", self._current_loss))
        iter_used = int(getattr(result, "nit", iter_count) or iter_count)
        rel_improvement = 0.0
        if initial_loss > 0:
            rel_improvement = abs(initial_loss - final_loss) / max(initial_loss, 1e-12)

        r2_total, r2_avg = self.compute_r2(getattr(result, "x", initial_guess))
        result.r2_total = r2_total
        result.r2_avg = r2_avg
        result.r2_target = r2_target

        criteria_met = []
        max_loss_exceeded = False
        if max_loss is not None and final_loss > max_loss:
            max_loss_exceeded = True
        if r2_target is not None and r2_total >= r2_target:
            criteria_met.append("r2")
        if abs_tol is not None and final_loss <= abs_tol:
            criteria_met.append("abs_tol")
        if rel_tol is not None and rel_improvement >= rel_tol:
            criteria_met.append("rel_tol")

        converged = bool(criteria_met) and not max_loss_exceeded
        if had_exception:
            converged = False
        if converged:
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
