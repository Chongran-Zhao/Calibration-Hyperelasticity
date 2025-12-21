import numpy as np
from scipy.optimize import minimize, differential_evolution
from utils import get_stress_component
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
            mean_stress = np.mean(stress_exp)
            ss_tot = np.sum((stress_exp - mean_stress)**2)
            
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
            stress_exp = dataset['stress_exp']
            ss_tot = self.ss_tot_list[i]
            
            model_stresses = []
            for F in f_list:
                # Use kinematic solver to get P_11 etc.
                P_tensor = self.solver.get_1st_PK_stress(F, params_dict)
                val = get_stress_component(P_tensor, mode)
                model_stresses.append(val)
            
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
        def callback_minimize(xk):
            pbar.set_postfix(Loss=f"{self._current_loss:.6f}")
            pbar.update(1)

        # Callback wrapper for differential_evolution
        def callback_de(xk, convergence):
            pbar.set_postfix(Loss=f"{self._current_loss:.6f}", Conv=f"{convergence:.4f}")
            pbar.update(1)

        result = None
        
        try:
            if method == 'Differential Evolution':
                # Global Optimization
                # DE requires finite bounds. Fill None with heuristic large range.
                safe_bounds = []
                if bounds:
                    for i, b in enumerate(bounds):
                        low = b[0] if b[0] is not None else 0.001
                        high = b[1] if b[1] is not None else 100.0 # Heuristic upper bound
                        safe_bounds.append((low, high))
                else:
                    # Fallback if absolutely no bounds given
                    safe_bounds = [(0.001, 100.0)] * len(initial_guess)

                result = differential_evolution(
                    func=self._objective_function,
                    bounds=safe_bounds,
                    callback=callback_de,
                    maxiter=100,      # Max generations
                    popsize=15,       # Population size multiplier
                    disp=False,
                    polish=True       # Polish with L-BFGS-B at the end
                )
                
            else:
                # Local Minimization (L-BFGS-B, Nelder-Mead, Powell, TNC, SLSQP)
                # Some methods (Nelder-Mead) technically don't support bounds in older Scipy,
                # but 'minimize' wrapper usually handles it or warns.
                
                # Filter bounds for methods that support them
                supports_bounds = ['L-BFGS-B', 'TNC', 'SLSQP', 'Powell']
                use_bounds = bounds if method in supports_bounds else None
                
                result = minimize(
                    fun=self._objective_function,
                    x0=initial_guess,
                    method=method,
                    bounds=use_bounds,
                    callback=callback_minimize,
                    options={'maxiter': 2000, 'disp': False}
                )

        except Exception as e:
            # Catch any numerical errors (e.g. during Jacobian calc) and return a failed state
            print(f"\nOptimization Error: {e}")
            from scipy.optimize import OptimizeResult
            result = OptimizeResult(success=False, message=str(e), fun=self._current_loss, x=initial_guess)
        
        finally:
            pbar.close()
        
        return result