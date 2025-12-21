import sympy as sp
import numpy as np

class Kinematics:
    """
    Kinematics class for hyperelastic materials.
    Automatically detects model type (invariant vs stretch) from function tags.
    """

    def __init__(self, energy_function, param_names):
        """
        Args:
            energy_function: Function from MaterialModels (must have @register_model tags).
            param_names: List of parameter names.
        """
        self.energy_function = energy_function
        self.param_names_ordered = param_names
        
        # 1. Detect Model Type from Tags
        if not hasattr(energy_function, 'model_type'):
            raise ValueError(f"Function {energy_function.__name__} missing 'model_type' tag.")
            
        tag_type = energy_function.model_type
        if tag_type == 'invariant_based':
            self.model_type = 'invariant'
        elif tag_type == 'stretch_based':
            self.model_type = 'stretch'
        else:
            raise ValueError(f"Unknown model_type tag: {tag_type}")

        # 2. Generate Parameter Symbols
        self.param_symbols = {name: sp.Symbol(name) for name in param_names}
        self.param_symbols_list = [self.param_symbols[name] for name in param_names]

        # 3. Prepare Derivatives based on Model Type
        if self.model_type == 'invariant':
            self._prepare_invariant_derivatives()
        elif self.model_type == 'stretch':
            self._prepare_stretch_derivatives()

    def _prepare_invariant_derivatives(self):
        """
        Pre-calculates scalar derivatives dPsi/dI1 and dPsi/dI2.
        """
        I1, I2 = sp.symbols('I1 I2')
        
        # Try calling with (I1, params), fallback to (I1, I2, params)
        try:
            psi_expr = self.energy_function(I1, self.param_symbols)
        except TypeError:
            psi_expr = self.energy_function(I1, I2, self.param_symbols)
            
        dPsi_dI1_sym = sp.diff(psi_expr, I1)
        dPsi_dI2_sym = sp.diff(psi_expr, I2)
        
        inputs = [I1, I2] + self.param_symbols_list
        self.calc_dPsi_dI1 = sp.lambdify(inputs, dPsi_dI1_sym, modules='numpy')
        self.calc_dPsi_dI2 = sp.lambdify(inputs, dPsi_dI2_sym, modules='numpy')

    def _prepare_stretch_derivatives(self):
        """
        Pre-calculates scalar derivatives dPsi/dlambda_i.
        """
        l1, l2, l3 = sp.symbols('lambda_1 lambda_2 lambda_3')
        
        psi_expr = self.energy_function(l1, l2, l3, self.param_symbols)
        
        dPsi_dl1_sym = sp.diff(psi_expr, l1)
        dPsi_dl2_sym = sp.diff(psi_expr, l2)
        dPsi_dl3_sym = sp.diff(psi_expr, l3)
        
        inputs = [l1, l2, l3] + self.param_symbols_list
        self.calc_dPsi_dl1 = sp.lambdify(inputs, dPsi_dl1_sym, modules='numpy')
        self.calc_dPsi_dl2 = sp.lambdify(inputs, dPsi_dl2_sym, modules='numpy')
        self.calc_dPsi_dl3 = sp.lambdify(inputs, dPsi_dl3_sym, modules='numpy')

    def get_2nd_PK_stress(self, F, params):
        p_vals = [params[name] for name in self.param_names_ordered]
        
        if self.model_type == 'invariant':
            C = F.T @ F
            I1 = np.trace(C)
            C2 = C @ C
            I2 = 0.5 * (I1**2 - np.trace(C2))
            
            Identity = np.eye(3)
            psi1 = self.calc_dPsi_dI1(I1, I2, *p_vals)
            psi2 = self.calc_dPsi_dI2(I1, I2, *p_vals)
            
            S_hyper = 2.0 * ((psi1 + I1 * psi2) * Identity - psi2 * C)
            
        elif self.model_type == 'stretch':
            l1, l2, l3 = F[0,0], F[1,1], F[2,2]
            
            s1 = self.calc_dPsi_dl1(l1, l2, l3, *p_vals)
            s2 = self.calc_dPsi_dl2(l1, l2, l3, *p_vals)
            s3 = self.calc_dPsi_dl3(l1, l2, l3, *p_vals)
            
            S_hyper = np.diag([s1/l1, s2/l2, s3/l3])
            
        Sigma_hyper = F @ S_hyper @ F.T
        p = Sigma_hyper[2, 2]
        
        C = F.T @ F
        C_inv = np.linalg.inv(C)
        S_total = S_hyper - p * C_inv
        
        return S_total

    def get_1st_PK_stress(self, F, params):
        S = self.get_2nd_PK_stress(F, params)
        return F @ S

    def get_Cauchy_stress(self, F, params):
        S = self.get_2nd_PK_stress(F, params)
        return F @ S @ F.T