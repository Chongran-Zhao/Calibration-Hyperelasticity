import sympy as sp
import numpy as np

class Kinematics:
    """
    Kinematics class for hyperelastic materials.
    Calculates stresses using the Chain Rule method.
    Separates scalar derivatives (physics) from tensor assembly (geometry).
    """

    def __init__(self, energy_function, param_names, model_type='invariant'):
        """
        Args:
            energy_function: Function from material_models.
            param_names: List of parameter names.
            model_type: 'invariant' or 'stretch'.
        """
        self.energy_function = energy_function
        self.model_type = model_type
        self.param_names_ordered = param_names
        
        # 1. Generate Parameter Symbols
        self.param_symbols = {name: sp.Symbol(name) for name in param_names}
        self.param_symbols_list = [self.param_symbols[name] for name in param_names]

        # 2. Prepare Derivatives based on Model Type
        if model_type == 'invariant':
            self._prepare_invariant_derivatives()
        elif model_type == 'stretch':
            self._prepare_stretch_derivatives()
        else:
            raise ValueError("Unsupported model_type. Use 'invariant' or 'stretch'.")

    def _prepare_invariant_derivatives(self):
        """
        Pre-calculates scalar derivatives dPsi/dI1 and dPsi/dI2.
        """
        # Define scalar symbols for invariants
        I1, I2 = sp.symbols('I1 I2')
        
        # Get Energy Expression Psi(I1, I2)
        # Try calling with (I1, params), fallback to (I1, I2, params)
        try:
            psi_expr = self.energy_function(I1, self.param_symbols)
        except TypeError:
            psi_expr = self.energy_function(I1, I2, self.param_symbols)
            
        # Symbolic Differentiation (Scalar level, very fast)
        dPsi_dI1_sym = sp.diff(psi_expr, I1)
        dPsi_dI2_sym = sp.diff(psi_expr, I2)
        
        # Compile functions
        # Inputs: (I1_val, I2_val, p1, p2...)
        inputs = [I1, I2] + self.param_symbols_list
        self.calc_dPsi_dI1 = sp.lambdify(inputs, dPsi_dI1_sym, modules='numpy')
        self.calc_dPsi_dI2 = sp.lambdify(inputs, dPsi_dI2_sym, modules='numpy')

    def _prepare_stretch_derivatives(self):
        """
        Pre-calculates scalar derivatives dPsi/dlambda_i.
        """
        # Define symbols for principal stretches
        l1, l2, l3 = sp.symbols('lambda_1 lambda_2 lambda_3')
        
        # Get Energy Expression Psi(l1, l2, l3)
        psi_expr = self.energy_function(l1, l2, l3, self.param_symbols)
        
        # Derivatives
        dPsi_dl1_sym = sp.diff(psi_expr, l1)
        dPsi_dl2_sym = sp.diff(psi_expr, l2)
        dPsi_dl3_sym = sp.diff(psi_expr, l3)
        
        # Compile functions
        inputs = [l1, l2, l3] + self.param_symbols_list
        self.calc_dPsi_dl1 = sp.lambdify(inputs, dPsi_dl1_sym, modules='numpy')
        self.calc_dPsi_dl2 = sp.lambdify(inputs, dPsi_dl2_sym, modules='numpy')
        self.calc_dPsi_dl3 = sp.lambdify(inputs, dPsi_dl3_sym, modules='numpy')

    def get_2nd_PK_stress(self, F, params):
        """
        Computes S using analytic tensor formulas.
        """
        # Unpack parameter values
        p_vals = [params[name] for name in self.param_names_ordered]
        
        if self.model_type == 'invariant':
            # 1. Calculate Kinematics
            C = F.T @ F
            I1 = np.trace(C)
            C2 = C @ C
            I2 = 0.5 * (I1**2 - np.trace(C2))
            
            # 2. Evaluate Scalar Response Coefficients
            # Handle cases where derivatives return scalar 0 (not array)
            psi1 = self.calc_dPsi_dI1(I1, I2, *p_vals)
            psi2 = self.calc_dPsi_dI2(I1, I2, *p_vals)
            
            # 3. Assemble Stress Tensor (Standard Formula)
            # S_iso = 2 * ( (dPsi/dI1 + I1*dPsi/dI2)*I - dPsi/dI2*C )
            Identity = np.eye(3)
            S_hyper = 2.0 * ((psi1 + I1 * psi2) * Identity - psi2 * C)
            
        elif self.model_type == 'stretch':
            # Assumes F is diagonal (Principal Stretches) for simplicity in calibration
            # If F is not diagonal, eigenvalues would be needed.
            l1, l2, l3 = F[0,0], F[1,1], F[2,2]
            
            # Evaluate derivatives
            s1 = self.calc_dPsi_dl1(l1, l2, l3, *p_vals)
            s2 = self.calc_dPsi_dl2(l1, l2, l3, *p_vals)
            s3 = self.calc_dPsi_dl3(l1, l2, l3, *p_vals)
            
            # S_i = (1/lambda_i) * dPsi/dlambda_i
            # Avoid divide by zero check handled by data validity
            S_hyper = np.diag([s1/l1, s2/l2, s3/l3])
            
        # --- Incompressibility Handling (Common for both) ---
        # Solve for p using sigma_33 = 0 condition
        # Sigma_hyper = F * S_hyper * F.T
        Sigma_hyper = F @ S_hyper @ F.T
        p = Sigma_hyper[2, 2]
        
        # S_total = S_hyper - p * C^-1
        C = F.T @ F # Recompute or reuse
        C_inv = np.linalg.inv(C)
        S_total = S_hyper - p * C_inv
        
        return S_total

    def get_1st_PK_stress(self, F, params):
        S = self.get_2nd_PK_stress(F, params)
        return F @ S

    def get_Cauchy_stress(self, F, params):
        S = self.get_2nd_PK_stress(F, params)
        return F @ S @ F.T
# EOF
