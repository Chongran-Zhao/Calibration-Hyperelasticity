import numpy as np
from scipy.optimize import minimize
from utils import get_stress_component

class MaterialOptimizer:
    """
    Optimizer for calibrating hyperelastic material parameters.
    """

    def __init__(self, kinematics_solver, experimental_data):
        """
        Args:
            kinematics_solver: Instance of Kinematics class.
            experimental_data: List of dicts (loaded from utils.py).
        """
        self.solver = kinematics_solver
        self.data = experimental_data
        self.param_names = kinematics_solver.param_names_ordered

    def _objective_function(self, params_array):
        """
        Loss function to minimize (Sum of Squared Errors).
        """
        # Convert array back to dict
        params_dict = dict(zip(self.param_names, params_array))
        
        total_error = 0.0
        
        for dataset in self.data:
            mode = dataset['mode']
            f_list = dataset['F_list']
            stress_exp = dataset['stress_exp']
            
            model_stresses = []
            for F in f_list:
                # 1. Calculate P tensor
                P_tensor = self.solver.get_1st_PK_stress(F, params_dict)
                
                # 2. Extract scalar component (Using helper from utils)
                val = get_stress_component(P_tensor, mode)
                model_stresses.append(val)
            
            model_stresses = np.array(model_stresses)
            
            # 3. Calculate Squared Error
            diff = model_stresses - stress_exp
            total_error += np.sum(diff**2)
            
        return total_error

    def fit(self, initial_guess, bounds=None):
        """
        Perform the optimization.
        """
        print("Starting optimization...")
        print(f"Parameters to fit: {self.param_names}")
        
        result = minimize(
            fun=self._objective_function,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'maxiter': 1000}
        )
        
        return result
# EOF
