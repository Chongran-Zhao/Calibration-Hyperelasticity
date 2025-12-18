import numpy as np
from scipy.optimize import minimize
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
        Equivalent to minimizing Sum(SS_res / SS_tot).
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
                P_tensor = self.solver.get_1st_PK_stress(F, params_dict)
                val = get_stress_component(P_tensor, mode)
                model_stresses.append(val)
            
            model_stresses = np.array(model_stresses)
            
            diff = model_stresses - stress_exp
            ss_res = np.sum(diff**2)
            
            total_normalized_error += ss_res / ss_tot
            
        self._current_loss = total_normalized_error
        return total_normalized_error

    def fit(self, initial_guess, bounds=None):
        """
        Perform the optimization with a visual progress bar.
        """
        print("Starting optimization (Metric: Sum of 1-R^2)...")
        
        max_iter = 1000
        
        with tqdm(total=max_iter, desc="  Fitting", unit="iter", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
            
            def progress_callback(xk):
                # Display current Loss value
                pbar.set_postfix(Loss=f"{self._current_loss:.6f}") # Changed Cost to Loss
                pbar.update(1)

            result = minimize(
                fun=self._objective_function,
                x0=initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                callback=progress_callback,
                options={'maxiter': max_iter, 'disp': False}
            )
            
            if result.success:
                pbar.total = pbar.n
                pbar.refresh()
        
        return result
# EOF
