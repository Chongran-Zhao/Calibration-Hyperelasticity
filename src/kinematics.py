import sympy as sp
import numpy as np

class Kinematics:
    """
    Kinematics class for hyperelastic materials.
    Calculates stresses based on the Strain Energy Density (Psi).
    """

    def __init__(self, energy_function, param_names, model_type='invariant'):
        """
        Args:
            energy_function: Function from material_models.
            param_names: List of parameter names as strings (e.g., ['C1', 'C2']).
                         The class will automatically create symbols for them.
            model_type: 'invariant' or 'stretch'.
        """
        self.energy_function = energy_function
        self.model_type = model_type
        
        # 1. Automatically generate SymPy symbols from the list of names
        # e.g., ['C1'] -> {'C1': Symbol('C1')}
        self.param_symbols = {name: sp.Symbol(name) for name in param_names}
        
        # Keep an ordered list for lambdify (compilation) later
        self.param_names_ordered = param_names
        self.param_symbols_ordered = [self.param_symbols[name] for name in param_names]

        # 2. Define Kinematic Symbols
        # F is a 3x3 Matrix Symbol
        self.F_sym = sp.MatrixSymbol('F', 3, 3)
        self.F_mat = sp.Matrix(self.F_sym) 
        
        # Right Cauchy-Green Tensor: C = F.T * F
        self.C_sym = self.F_mat.T * self.F_mat
        
        # 3. Construct the Energy Expression (Psi)
        if model_type == 'invariant':
            I1 = sp.trace(self.C_sym)
            I2 = 0.5 * (I1**2 - sp.trace(self.C_sym * self.C_sym))
            
            # Try calling with (I1, params) first, then (I1, I2, params)
            try:
                self.psi_expr = energy_function(I1, self.param_symbols)
            except TypeError:
                self.psi_expr = energy_function(I1, I2, self.param_symbols)
                
        elif model_type == 'stretch':
            # Assume diagonal C for simple differentiation of stretches
            l1 = sp.sqrt(self.C_sym[0, 0])
            l2 = sp.sqrt(self.C_sym[1, 1])
            l3 = sp.sqrt(self.C_sym[2, 2])
            self.psi_expr = energy_function(l1, l2, l3, self.param_symbols)
        else:
            raise ValueError("Unsupported model_type. Use 'invariant' or 'stretch'.")

        # 4. Compute Hyperelastic Second PK Stress (S_hyper) symbolically
        # Formula: S = 2 * d(Psi) / dC
        self.S_hyper_sym = 2 * self._diff_matrix(self.psi_expr, self.C_sym)

        # 5. Compile the function for fast numerical execution
        # The compiled function takes (F_matrix, param_value_1, param_value_2, ...)
        self.calc_S_hyper_fast = sp.lambdify(
            [self.F_sym] + self.param_symbols_ordered, 
            self.S_hyper_sym, 
            modules='numpy'
        )

    def _diff_matrix(self, scalar_func, matrix_var):
        """
        Differentiates a scalar function w.r.t each component of a matrix.
        Ensures the output is a matrix of the same shape.
        """
        rows, cols = matrix_var.shape
        res = sp.zeros(rows, cols)
        for i in range(rows):
            for j in range(cols):
                res[i, j] = sp.diff(scalar_func, matrix_var[i, j])
        return res

    def get_2nd_PK_stress(self, F, params):
        """
        Compute Second Piola-Kirchhoff Stress (S).
        Automatically solves for hydrostatic pressure p assuming sigma_33 = 0.
        
        Args:
            F: (3,3) Deformation Gradient (numpy array)
            params: Dictionary of parameter values (e.g., {'C1': 0.5})
        """
        # 1. Extract values in the correct order matches param_names
        p_values = [params[name] for name in self.param_names_ordered]
        
        # 2. Calculate Hyperelastic part of S
        S_hyper = np.array(self.calc_S_hyper_fast(F, *p_values))
        
        # 3. Solve for Hydrostatic Pressure (p)
        # Condition: Sigma_33 = 0
        # Relation: Sigma_hyper = F * S_hyper * F.T
        # p = Sigma_hyper_33 (since Sigma_total_33 = Sigma_hyper_33 - p = 0)
        
        Sigma_hyper = F @ S_hyper @ F.T
        p = Sigma_hyper[2, 2] 
        
        # 4. Apply pressure correction to S
        # S_total = S_hyper - p * C^-1
        C = F.T @ F
        C_inv = np.linalg.inv(C)
        S_total = S_hyper - p * C_inv
        
        return S_total

    def get_1st_PK_stress(self, F, params):
        """
        Compute First Piola-Kirchhoff Stress (P).
        P = F * S
        """
        S = self.get_2nd_PK_stress(F, params)
        P = F @ S
        return P

    def get_Cauchy_stress(self, F, params):
        """
        Compute Cauchy Stress (Sigma).
        Sigma = F * S * F.T
        """
        S = self.get_2nd_PK_stress(F, params)
        Sigma = F @ S @ F.T
        return Sigma
