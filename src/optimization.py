import numpy as np
from scipy.optimize import minimize
from utils import get_stress_components
from tqdm import tqdm

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
            
            model_stresses = []
            for F in f_list:
                if stress_type == 'cauchy':
                    stress_tensor = self.solver.get_Cauchy_stress(F, params_dict)
                else:
                    stress_tensor = self.solver.get_1st_PK_stress(F, params_dict)
                components = get_stress_components(stress_tensor, mode)
                if np.ndim(stress_exp) == 1:
                    model_stresses.append(components[0])
                else:
                    model_stresses.append(components[:stress_exp.shape[1]])
            
            model_stresses = np.array(model_stresses)
            
            # Simple Sum of Squared Residuals
            diff = model_stresses - stress_exp
            ss_res = np.sum(diff**2)
            
            # Normalize by Variance (1 - R^2 style)
            total_normalized_error += ss_res / ss_tot
            
        self._current_loss = total_normalized_error
        return total_normalized_error

    def fit(self, initial_guess, bounds=None, method='L-BFGS-B'):
        """
        Perform the optimization using the selected method.
        
        Args:
            initial_guess (list): Starting parameters.
            bounds (list): Parameter bounds [(min, max), ...].
            method (str): Optimization algorithm.
        """
        print(f"Starting optimization using {method}...")
        
        # --- Progress Bar Setup ---
        # Different methods iterate differently, so we use a generic counter
        pbar = tqdm(desc=f"  Fitting", unit="iter", 
                    bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}{postfix}]")
        
        # Callback wrapper for local minimizers (minimize)
        def callback_minimize(xk, *args):
            pbar.set_postfix(Loss=f"{self._current_loss:.6f}")
            pbar.update(1)

        # Callback wrapper for differential_evolution
        def callback_de(xk, convergence):
            pbar.set_postfix(Loss=f"{self._current_loss:.6f}", Conv=f"{convergence:.4f}")
            pbar.update(1)

        result = None
        supported_methods = {'L-BFGS-B', 'trust-constr', 'CG', 'Newton-CG'}

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

        try:
            if method not in supported_methods:
                raise ValueError(f"Unsupported method: {method}")

            use_bounds = bounds if method in {'L-BFGS-B', 'trust-constr'} else None
            jac = numeric_grad if method == 'Newton-CG' else None

            result = minimize(
                fun=self._objective_function,
                x0=initial_guess,
                method=method,
                bounds=use_bounds,
                jac=jac,
                callback=callback_minimize,
                options={'maxiter': 2000, 'disp': False}
            )

        except Exception as e:
            print(f"\nOptimization Error: {e}")
            from scipy.optimize import OptimizeResult
            result = OptimizeResult(success=False, message=str(e), fun=self._current_loss, x=initial_guess)

        finally:
            pbar.close()

        return result
