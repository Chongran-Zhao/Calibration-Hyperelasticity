import numpy as np
from scipy.optimize import minimize
from utils import get_stress_component
from tqdm import tqdm  # Import progress bar library

class MaterialOptimizer:
    """
    Optimizer for calibrating hyperelastic material parameters.
    """

    def __init__(self, kinematics_solver, experimental_data):
        self.solver = kinematics_solver
        self.data = experimental_data
        self.param_names = kinematics_solver.param_names_ordered
        self._current_loss = float('inf') # Store loss for display

    def _objective_function(self, params_array):
        """
        Loss function to minimize (Sum of Squared Errors).
        """
        params_dict = dict(zip(self.param_names, params_array))
        total_error = 0.0
        
        for dataset in self.data:
            mode = dataset['mode']
            f_list = dataset['F_list']
            stress_exp = dataset['stress_exp']
            
            # Vectorized calculation usually requires deeper refactoring, 
            # keeping loop for safety with symbolic tensor logic.
            model_stresses = []
            for F in f_list:
                P_tensor = self.solver.get_1st_PK_stress(F, params_dict)
                val = get_stress_component(P_tensor, mode)
                model_stresses.append(val)
            
            model_stresses = np.array(model_stresses)
            diff = model_stresses - stress_exp
            total_error += np.sum(diff**2)
            
        # Cache the loss value so the callback can display it
        self._current_loss = total_error
        return total_error

    def fit(self, initial_guess, bounds=None):
        """
        Perform the optimization with a visual progress bar.
        """
        print("Starting optimization...")
        
        max_iter = 1000
        
        # Initialize tqdm progress bar
        with tqdm(total=max_iter, desc="  Fitting", unit="iter", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
            
            # Define a callback that updates the progress bar
            # scipy passes the current parameter vector (xk) to the callback
            def progress_callback(xk):
                pbar.set_postfix(SSE=f"{self._current_loss:.4f}")
                pbar.update(1)

            result = minimize(
                fun=self._objective_function,
                x0=initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                callback=progress_callback,
                options={'maxiter': max_iter, 'disp': False} # Turn off scipy's internal print
            )
            
            # Ensure bar is full if converged early
            if result.success:
                pbar.total = pbar.n
                pbar.refresh()
        
        return result

# EOF
