import sympy as sp
from generalized_strains import GeneralizedStrains
from utils import inv_Langevin_Kroger

# =============================================================================
# Global Model Formula Dictionary
# =============================================================================
MODEL_FORMULAS = {
    # Invariant Models
    "NeoHookean": r"Psi = C1 * (I1 - 3)",
    "MooneyRivlin": r"Psi = C1 * (I1 - 3) + C2 * (I2 - 3)",
    "Yeoh": r"Psi = C1*(I1-3) + C2*(I1-3)^2 + C3*(I1-3)^3",
    "ArrudaBoyce": (
        r"Psi = mu * N * [ lambda_r * beta + ln( beta / sinh(beta) ) ]" + "\n" +
        r"      where lambda_r = sqrt(I1 / 3N), beta = L^(-1)(lambda_r)" + "\n" +
        r"      (Using Kroger's Pade approximation for inverse Langevin)"
    ),
    
    # Stretch-Based Models
    "Ogden": r"Psi = Sum [ (mu_i / alpha_i) * (lambda_1^alpha_i + lambda_2^alpha_i + lambda_3^alpha_i - 3) ]",
    "Hill": r"Psi = Sum [ C_i * ( E(lambda_1)^2 + E(lambda_2)^2 + E(lambda_3)^2 ) ] (Generalized Strain Energy)"
}

def print_model_formula(model_name):
    """
    Prints a clean, human-readable LaTeX-like formula for the model.
    This replaces the individual print_info methods in classes.
    """
    print("-" * 60)
    print(f"Model Formula ({model_name}):")
    print(MODEL_FORMULAS.get(model_name, "No formula display available for this model."))
    print("-" * 60)


# =============================================================================
# Model Classes
# =============================================================================

class InvariantModels:
    """
    Hyperelastic models based on invariants (I1, I2).
    Inputs are SymPy symbols for invariants and a dictionary for parameters.
    Assumes incompressibility (J=1).
    """

    @staticmethod
    def NeoHookean(I1, params):
        """
        Neo-Hookean Model.
        Psi = C1 * (I1 - 3)
        """
        return params['C1'] * (I1 - 3)

    @staticmethod
    def MooneyRivlin(I1, I2, params):
        """
        Mooney-Rivlin Model.
        Psi = C1 * (I1 - 3) + C2 * (I2 - 3)
        """
        return params['C1'] * (I1 - 3) + params['C2'] * (I2 - 3)

    @staticmethod
    def Yeoh(I1, params):
        """
        Yeoh Model (3rd order).
        Psi = C1*(I1-3) + C2*(I1-3)^2 + C3*(I1-3)^3
        """
        return (params['C1'] * (I1 - 3) + 
                params['C2'] * (I1 - 3)**2 + 
                params['C3'] * (I1 - 3)**3)

    @staticmethod
    def ArrudaBoyce(I1, params):
        """
        Arruda-Boyce (8-chain model) using Kröger's approximation.
        
        Psi = mu * N * [ lambda_r * beta + ln( beta / sinh(beta) ) ]
        where:
          N: Number of rigid links (N_chain)
          lambda_r = sqrt( I1 / (3N) )
          beta = inv_Langevin(lambda_r)
        """
        mu = params['mu']
        N = params['N'] # Corresponds to N_chain
        
        # 1. Calculate relative average network stretch (lambda_r)
        # lambda_r = sqrt( I1 / (3N) )
        lambda_r = sp.sqrt(I1 / (3.0 * N))
        
        # 2. Calculate beta using the helper from utils.py
        beta = inv_Langevin_Kroger(lambda_r)
        
        # 3. Calculate Energy Density
        psi_chain = lambda_r * beta + sp.log(beta / sp.sinh(beta))
        Psi = mu * N * psi_chain
        
        return Psi


class StretchBasedModels:
    """
    Hyperelastic models based on principal stretches (lambda_1, lambda_2, lambda_3).
    Assumes incompressibility (lambda_1 * lambda_2 * lambda_3 = 1).
    """

    @staticmethod
    def Ogden(lambda_1, lambda_2, lambda_3, params):
        """
        Ogden Model (N-terms).
        Psi = sum( (mu_i / alpha_i) * (lambda_1^alpha_i + lambda_2^alpha_i + lambda_3^alpha_i - 3) )
        """
        mus = params['mu']
        alphas = params['alpha']
        
        psi = 0
        # mus and alphas should be lists of symbols
        for mu, alpha in zip(mus, alphas):
            psi += (mu / alpha) * (lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3)
            
        return psi

    @staticmethod
    def Hill(lambda_1, lambda_2, lambda_3, params):
        """
        Smart Generalized Hill Model.
        Automatically constructs energy based on grouped parameters.
        """
        psi = 0
        
        for strain_type, p_dict in params.items():
            if not hasattr(GeneralizedStrains, strain_type):
                raise ValueError(f"Unknown strain type: {strain_type}")
            
            strain_func = getattr(GeneralizedStrains, strain_type)
            
            if 'C' not in p_dict:
                raise ValueError(f"Missing coefficient 'C' for {strain_type}")
            
            C_list = p_dict['C']
            num_terms = len(C_list)
            
            param_keys = [k for k in p_dict.keys() if k != 'C']
            
            for i in range(num_terms):
                C = C_list[i]
                current_params = {}
                for key in param_keys:
                    current_params[key] = p_dict[key][i]
                
                E1 = strain_func(lambda_1, current_params)
                E2 = strain_func(lambda_2, current_params)
                E3 = strain_func(lambda_3, current_params)
                
                psi += C * (E1**2 + E2**2 + E3**2)
                
        return psi
# EOF
