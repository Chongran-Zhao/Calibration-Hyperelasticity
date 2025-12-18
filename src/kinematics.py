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
            model_type: 'invariant' or 'stretch'.
        """
        self.energy_function = energy_function
        self.model_type = model_type
        
        # 1. Generate SymPy symbols for parameters
        self.param_symbols = {name: sp.Symbol(name) for name in param_names}
        self.param_names_ordered = param_names
        self.param_symbols_ordered = [self.param_symbols[name] for name in param_names]

        # 2. Define Kinematic Symbols
        # F is the input Matrix Symbol (for lambdify input)
        self.F_sym = sp.MatrixSymbol('F', 3, 3)
        self.F_mat = sp.Matrix(self.F_sym) 
        
        # --- CRITICAL FIX FOR DERIVATIVE ---
        # To differentiate Psi w.r.t C, C must be composed of independent symbols first.
        # We cannot differentiate w.r.t an expression like (F.T * F).
        
        # Define a temporary auxiliary C matrix with independent components
        self.C_aux = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f'C_aux_{i}{j}'))
        
        # 3. Construct Energy Expression (Psi) using C_aux
        if model_type == 'invariant':
            # Calculate invariants from the auxiliary C
            I1 = sp.trace(self.C_aux)
            I2 = 0.5 * (I1**2 - sp.trace(self.C_aux * self.C_aux))
            
            try:
                self.psi_expr = energy_function(I1, self.param_symbols)
            except TypeError:
                self.psi_expr = energy_function(I1, I2, self.param_symbols)
                
        elif model_type == 'stretch':
            # Assume diagonal C for simple differentiation of stretches
            l1 = sp.sqrt(self.C_aux[0, 0])
            l2 = sp.sqrt(self.C_aux[1, 1])
            l3 = sp.sqrt(self.C_aux[2, 2])
            self.psi_expr = energy_function(l1, l2, l3, self.param_symbols)
        else:
            raise ValueError("Unsupported model_type. Use 'invariant' or 'stretch'.")

        # 4. Compute S_hyper symbolic using C_aux
        # S = 2 * d(Psi) / dC
        # Now we can differentiate because C_aux components are simple symbols
        S_aux = 2 * self._diff_matrix(self.psi_expr, self.C_aux)

        # 5. Substitute C_aux back with (F.T * F)
        # This expresses S in terms of our input F, which is what we need for calculation
        C_calc = self.F_mat.T * self.F_mat
        
        # Create a substitution dictionary: {C_aux_00: (F.T*F)[0,0], ...}
        replacements = {self.C_aux[i, j]: C_calc[i, j] for i in range(3) for j in range(3)}
        
        # Perform substitution
        self.S_hyper_sym = S_aux.subs(replacements)

        # 6. Compile the function
        self.calc_S_hyper_fast = sp.lambdify(
            [self.F_sym] + self.param_symbols_ordered, 
            self.S_hyper_sym, 
            modules='numpy'
        )

    def _diff_matrix(self, scalar_func, matrix_var):
        """
        Differentiates a scalar function w.r.t each component of a matrix.
        """
        rows, cols = matrix_var.shape
        res = sp.zeros(rows, cols)
        for i in range(rows):
            for j in range(cols):
                res[i, j] = sp.diff(scalar_func, matrix_var[i, j])
        return res

    def print_model_info(self):
        """
        Prints detailed information about the material model,
        including parameters and the symbolic energy expression.
        """
        model_name = self.energy_function.__name__
        num_params = len(self.param_names_ordered)

        print("="*60)
        print(f" Material Model Information")
        print("="*60)
        print(f" Model Name      : {model_name}")
        print(f" Model Type      : {self.model_type.capitalize()}")
        print(f" Parameter Count : {num_params}")
        print(f" Parameters      : {self.param_names_ordered}")
        print("-" * 60)
        print(f" Symbolic Energy Expression (Psi):")

        # Use SymPy's pretty printer for mathematical rendering
        sp.pprint(self.psi_expr, use_unicode=True)
        print("="*60 + "\n")

    def get_2nd_PK_stress(self, F, params):
        p_values = [params[name] for name in self.param_names_ordered]
        S_hyper = np.array(self.calc_S_hyper_fast(F, *p_values))
        
        # Solve for p (Sigma_33 = 0)
        Sigma_hyper = F @ S_hyper @ F.T
        p = Sigma_hyper[2, 2] 
        
        C = F.T @ F
        C_inv = np.linalg.inv(C)
        S_total = S_hyper - p * C_inv
        
        return S_total

    def get_1st_PK_stress(self, F, params):
        S = self.get_2nd_PK_stress(F, params)
        P = F @ S
        return P

    def get_Cauchy_stress(self, F, params):
        S = self.get_2nd_PK_stress(F, params)
        Sigma = F @ S @ F.T
        return Sigma
