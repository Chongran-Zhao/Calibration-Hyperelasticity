import sympy as sp
from generalized_strains import GeneralizedStrains

class InvariantModels:
    """
    Hyperelastic models based on invariants (I1, I2).
    Assumes incompressibility (J=1).
    """

    @staticmethod
    def NeoHookean(I1, params):
        """
        Psi = C1 * (I1 - 3)
        """
        return params['C1'] * (I1 - 3)

    @staticmethod
    def MooneyRivlin(I1, I2, params):
        """
        Psi = C1 * (I1 - 3) + C2 * (I2 - 3)
        """
        return params['C1'] * (I1 - 3) + params['C2'] * (I2 - 3)

    @staticmethod
    def Yeoh(I1, params):
        """
        Psi = C1*(I1-3) + C2*(I1-3)^2 + C3*(I1-3)^3
        """
        return (params['C1'] * (I1 - 3) + 
                params['C2'] * (I1 - 3)**2 + 
                params['C3'] * (I1 - 3)**3)

    @staticmethod
    def Gent(I1, params):
        """
        Psi = - (mu * Jm / 2) * ln(1 - (I1 - 3) / Jm)
        """
        return -0.5 * params['mu'] * params['Jm'] * sp.log(1 - (I1 - 3) / params['Jm'])
    
    @staticmethod
    def ArrudaBoyce(I1, params):
        """
        Arruda-Boyce (8-chain model).
        """
        mu = params['mu']
        lambda_m = params['lambda_m']
        
        C1 = 0.5 * mu
        C2 = (1 / (20 * lambda_m**2)) * mu
        C3 = (11 / (1050 * lambda_m**4)) * mu
        C4 = (19 / (7000 * lambda_m**6)) * mu
        C5 = (519 / (673750 * lambda_m**8)) * mu
        
        I1_bar = I1 - 3
        
        return (C1 * I1_bar + 
                C2 * (I1**2 - 9) + 
                C3 * (I1**3 - 27) + 
                C4 * (I1**4 - 81) + 
                C5 * (I1**5 - 243))


class StretchBasedModels:
    """
    Hyperelastic models based on principal stretches.
    Assumes incompressibility.
    """

    @staticmethod
    def Ogden(lambda_1, lambda_2, lambda_3, params):
        """
        Ogden Model.
        Psi = sum( mu_i * SethHill(lambda, alpha_i) )
        Note: Ogden usually doesn't square the term, it is linear sum of powers.
        """
        mus = params['mu']
        alphas = params['alpha']
        
        psi = 0
        for mu, alpha in zip(mus, alphas):
            # Map Ogden params to SethHill params
            term_params = {'m': alpha}
            # Ogden: mu * (lam^alpha - 1)/alpha
            term = mu * (GeneralizedStrains.SethHill(lambda_1, term_params) + 
                         GeneralizedStrains.SethHill(lambda_2, term_params) + 
                         GeneralizedStrains.SethHill(lambda_3, term_params))
            psi += term
            
        return psi

    @staticmethod
    def Hill(lambda_1, lambda_2, lambda_3, params):
        """
        Squared Generalized Hill Model.
        Psi = sum( C_i * [E(lambda_1)^2 + E(lambda_2)^2 + E(lambda_3)^2] )
        
        Args:
            params: Dict structured by Strain Type.
            
            Example structure for Darijani-Naghdabadi:
            {
                'DarijaniNaghdabadi': {
                    'C': [mu1],
                    'm': [1.5],
                    'n': [0.5]
                }
            }
        """
        psi = 0
        
        # Loop through each strain type requested
        for strain_type, p_dict in params.items():
            
            if not hasattr(GeneralizedStrains, strain_type):
                raise ValueError(f"Unknown strain type: {strain_type}")
            
            strain_func = getattr(GeneralizedStrains, strain_type)
            
            if 'C' not in p_dict:
                raise ValueError(f"Missing coefficient 'C' for {strain_type}")
            
            C_list = p_dict['C']
            num_terms = len(C_list)
            
            # Identify parameter keys (exclude 'C')
            param_keys = [k for k in p_dict.keys() if k != 'C']
            
            for i in range(num_terms):
                C = C_list[i]
                
                # Construct parameters for the current term
                current_params = {}
                for key in param_keys:
                    current_params[key] = p_dict[key][i]
                
                # Compute strains
                E1 = strain_func(lambda_1, current_params)
                E2 = strain_func(lambda_2, current_params)
                E3 = strain_func(lambda_3, current_params)
                
                # Add to total energy: C * (sum of SQUARED strains)
                psi += C * (E1**2 + E2**2 + E3**2)
                
        return psi

# EOF
