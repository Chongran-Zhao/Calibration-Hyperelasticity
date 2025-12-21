import sympy as sp
from generalized_strains import GeneralizedStrains, STRAIN_CONFIGS
from utils import inv_Langevin_Kroger

# =============================================================================
# Decorator for Model Tagging & Configuration
# =============================================================================
def register_model(model_type, category, formula_str="", param_names=None, initial_guess=None, bounds=None):
    def decorator(func):
        func.model_type = model_type
        func.category = category
        func.formula = formula_str
        func.param_names = param_names if param_names else []
        func.initial_guess = initial_guess if initial_guess else []
        func.bounds = bounds if bounds else []
        return func
    return decorator

# =============================================================================
# Unified Material Models Class
# =============================================================================
class MaterialModels:
    """
    Unified collection of hyperelastic material models.
    """

    # --- Invariant Based Models ---

    @staticmethod
    @register_model(
        model_type='invariant_based', 
        category='phenomenological',
        formula_str=r"Psi = C1 * (I1 - 3)",
        param_names=["C1"],
        initial_guess=[0.5],
        # Safe Bound: C1 > 0
        bounds=[(1e-6, None)]
    )
    def NeoHookean(I1, params):
        """Neo-Hookean Model"""
        return params['C1'] * (I1 - 3)

    @staticmethod
    @register_model(
        model_type='invariant_based', 
        category='phenomenological',
        formula_str=r"Psi = C1 * (I1 - 3) + C2 * (I2 - 3)",
        param_names=["C1", "C2"],
        initial_guess=[0.5, 0.1],
        # Safe Bound: C1 > 0. C2 allowed to be flexible or 0
        bounds=[(1e-6, None), (None, None)]
    )
    def MooneyRivlin(I1, I2, params):
        """Mooney-Rivlin Model"""
        return params['C1'] * (I1 - 3) + params['C2'] * (I2 - 3)

    @staticmethod
    @register_model(
        model_type='invariant_based', 
        category='phenomenological',
        formula_str=r"Psi = C1*(I1-3) + C2*(I1-3)^2 + C3*(I1-3)^3",
        param_names=["C1", "C2", "C3"],
        initial_guess=[0.5, -0.01, 0.001],
        # Safe Bound: C1 > 0. Higher order terms C2, C3 can be negative.
        bounds=[(1e-6, None), (None, None), (None, None)]
    )
    def Yeoh(I1, params):
        """Yeoh Model (3rd Order)"""
        return (params['C1'] * (I1 - 3) + 
                params['C2'] * (I1 - 3)**2 + 
                params['C3'] * (I1 - 3)**3)

    @staticmethod
    @register_model(
        model_type='invariant_based', 
        category='micromechanical',
        formula_str=r"Psi = mu * N * [ lambda_r * beta + ln( beta / sinh(beta) ) ]",
        param_names=["mu", "N"],
        initial_guess=[0.4, 10.0],
        # Safe Bound: mu > 0, N >= 1.0 (approximated, driver handles dynamic N limit)
        bounds=[(1e-6, None), (1.0, None)]
    )
    def ArrudaBoyce(I1, params):
        """Arruda-Boyce (8-Chain) Model"""
        mu = params['mu']
        N = params['N'] 
        lambda_r = sp.sqrt(I1 / (3.0 * N))
        beta = inv_Langevin_Kroger(lambda_r)
        return mu * N * (lambda_r * beta + sp.log(beta / sp.sinh(beta)))

    # --- Stretch Based Models ---

    @staticmethod
    @register_model(
        model_type='stretch_based', 
        category='phenomenological',
        formula_str=r"Psi = Sum [ (mu_i / alpha_i) * (lambda_1^alpha_i + ... - 3) ]",
        param_names=["mu", "alpha"],
        initial_guess=[0.5, 2.0],
        # Safe Bound: mu > 0
        bounds=[(1e-6, None), (None, None)]
    )
    def Ogden(lambda_1, lambda_2, lambda_3, params):
        """Ogden Model (1-term default)"""
        mus = params['mu']
        alphas = params['alpha']
        if not isinstance(mus, (list, tuple)): mus = [mus]
        if not isinstance(alphas, (list, tuple)): alphas = [alphas]
        
        psi = 0
        for mu, alpha in zip(mus, alphas):
            psi += (mu / alpha) * (lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3)
        return psi

    # --- Hill Model Factory ---

    @staticmethod
    def create_hill_model(strain_name):
        """
        Factory method to generate a specific Hill model function based on a generalized strain.
        This handles the parameter naming (mu, m1, n1 etc.) and metadata dynamically.
        Changed C1 -> mu for consistency.
        """
        if strain_name not in STRAIN_CONFIGS:
            raise ValueError(f"Unknown strain configuration: {strain_name}")
            
        # 1. Retrieve Config
        config = STRAIN_CONFIGS[strain_name]
        strain_params = config['params'] # e.g., ['m']
        strain_defaults = config['defaults']
        strain_bounds = config['bounds']
        
        # 2. Construct Full Parameter List
        # Changed C1 -> mu
        full_param_names = ['mu'] + [f"{p}1" for p in strain_params]
        full_initial_guess = [10.0] + strain_defaults
        
        # 3. Construct Full Bounds
        # Enforce mu > 0 using a small epsilon
        full_bounds = [(1e-6, None)] + strain_bounds

        # 4. Define the Closure Function
        def Hill_Dynamic(lambda_1, lambda_2, lambda_3, params):
            """Dynamically generated Hill Model"""
            # Changed C1 -> mu
            C = params['mu']
            
            current_params = {}
            for p_base in strain_params:
                p_key = f"{p_base}1"
                current_params[p_base] = params[p_key]
            
            method_name = strain_name.replace("-", "")
            strain_func = getattr(GeneralizedStrains, method_name)
            
            E1 = strain_func(lambda_1, current_params)
            E2 = strain_func(lambda_2, current_params)
            E3 = strain_func(lambda_3, current_params)
            
            return C * (E1**2 + E2**2 + E3**2)

        # 5. Attach Metadata manually
        Hill_Dynamic.__name__ = f"Hill_{strain_name}"
        Hill_Dynamic.model_type = 'stretch_based'
        Hill_Dynamic.category = 'phenomenological'
        Hill_Dynamic.formula = f"Psi = mu * (E_{strain_name}^2 + ...)"
        Hill_Dynamic.param_names = full_param_names
        Hill_Dynamic.initial_guess = full_initial_guess
        Hill_Dynamic.bounds = full_bounds
        
        return Hill_Dynamic

def print_model_info(model_func):
    """Helper to print model details from tags."""
    name = model_func.__name__
    m_type = getattr(model_func, 'model_type', 'Unknown')
    cat = getattr(model_func, 'category', 'Unknown')
    formula = getattr(model_func, 'formula', 'No formula')
    params = getattr(model_func, 'param_names', [])
    
    print("-" * 60)
    print(f"Model: {name}")
    print(f"Type: {m_type} | Category: {cat}")
    print(f"Parameters: {params}")
    print(f"Formula: {formula}")
    print("-" * 60)