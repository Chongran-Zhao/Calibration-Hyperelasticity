import sympy as sp
from generalized_strains import GeneralizedStrains, STRAIN_CONFIGS, STRAIN_FORMULAS
from utils import inv_Langevin_Kroger
from zhan_models import zhan_gaussian_pk1, zhan_nongaussian_pk1

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
        formula_str=r"\Psi = C_1 (I_1 - 3)",
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
        formula_str=r"\Psi = C_1 (I_1 - 3) + C_2 (I_2 - 3)",
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
        formula_str=r"\Psi = C_1 (I_1 - 3) + C_2 (I_1 - 3)^2 + C_3 (I_1 - 3)^3",
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
        formula_str=r"\Psi = \mu N \left[\lambda_r \beta + \ln\left(\beta / \sinh(\beta)\right)\right]",
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

    # --- Zhan Models (Custom Stress) ---

    @staticmethod
    @register_model(
        model_type='custom',
        category='micromechanical',
        formula_str=r"\Psi = \mu\left[(\lambda_1+\lambda_2+\lambda_3)^2 + 2(\lambda_1^2+\lambda_2^2+\lambda_3^2)\right]",
        param_names=["mu"],
        initial_guess=[0.5],
        bounds=[(1e-6, None)]
    )
    def ZhanGaussian(lambda_1, lambda_2, lambda_3, params):
        """Zhan Gaussian Model (custom PK1)"""
        mu = params['mu']
        return mu * ((lambda_1 + lambda_2 + lambda_3)**2 + 2.0 * (lambda_1**2 + lambda_2**2 + lambda_3**2))

    @staticmethod
    @register_model(
        model_type='custom',
        category='micromechanical',
        formula_str=r"\Psi = \mu\sqrt{N}\int_{\mathbb{S}^2} \left( \lambda_i \beta + \ln\left( \beta / \sinh(\beta) \right) \right) d\Omega",
        param_names=["mu", "N"],
        initial_guess=[0.5, 10.0],
        bounds=[(1e-6, None), (1.0, None)]
    )
    def ZhanNonGaussian(lambda_1, lambda_2, lambda_3, params):
        """Zhan non-Gaussian Model (custom PK1)"""
        mu = params['mu']
        N = params['N']
        return mu * sp.sqrt(N) * (lambda_1 + lambda_2 + lambda_3)

    # --- Stretch Based Models ---

    @staticmethod
    @register_model(
        model_type='stretch_based', 
        category='phenomenological',
        formula_str=r"\Psi = \sum_i \frac{\mu_i}{\alpha_i}\left(\lambda_1^{\alpha_i} + \lambda_2^{\alpha_i} + \lambda_3^{\alpha_i} - 3\right)",
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

    @staticmethod
    def create_ogden_model(n_terms):
        """
        Factory method to generate an Ogden model with n_terms.
        For n_terms == 1, use (mu, alpha). Otherwise use (mu1, alpha1, ...).
        """
        n_terms = int(n_terms)
        if n_terms < 1:
            raise ValueError("Ogden model must have at least one term.")

        if n_terms == 1:
            param_names = ["mu", "alpha"]
            initial_guess = [-0.5, -2.0]
            bounds = [(None, -1e-6), (None, -1e-6)]
        else:
            param_names = []
            initial_guess = []
            bounds = []
            for i in range(1, n_terms + 1):
                param_names.extend([f"mu{i}", f"alpha{i}"])
                if i == 1:
                    initial_guess.extend([-0.5, -2.0])
                    bounds.extend([(None, -1e-6), (None, -1e-6)])
                else:
                    initial_guess.extend([0.5 / i, 2.0 * i])
                    bounds.extend([(1e-6, None), (1e-6, None)])

        def Ogden_Dynamic(lambda_1, lambda_2, lambda_3, params):
            psi = 0
            for i in range(1, n_terms + 1):
                mu_key = "mu" if n_terms == 1 else f"mu{i}"
                alpha_key = "alpha" if n_terms == 1 else f"alpha{i}"
                mu = params[mu_key]
                alpha = params[alpha_key]
                psi += (mu / alpha) * (lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3)
            return psi

        Ogden_Dynamic.__name__ = f"Ogden_{n_terms}term"
        Ogden_Dynamic.model_type = 'stretch_based'
        Ogden_Dynamic.category = 'phenomenological'
        Ogden_Dynamic.formula = rf"\Psi = \sum_{{i=1}}^{{{n_terms}}} \frac{{\mu_i}}{{\alpha_i}}\left(\lambda_1^{{\alpha_i}} + \lambda_2^{{\alpha_i}} + \lambda_3^{{\alpha_i}} - 3\right)"
        Ogden_Dynamic.param_names = param_names
        Ogden_Dynamic.initial_guess = initial_guess
        Ogden_Dynamic.bounds = bounds

        return Ogden_Dynamic


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
        Hill_Dynamic.formula = rf"\Psi = \mu \sum_{{i=1}}^3 E_{{{strain_name}}}(\lambda_i)^2"
        Hill_Dynamic.strain_formula = STRAIN_FORMULAS.get(strain_name, "")
        Hill_Dynamic.param_names = full_param_names
        Hill_Dynamic.initial_guess = full_initial_guess
        Hill_Dynamic.bounds = full_bounds
        
        return Hill_Dynamic


# Attach custom stress functions for Zhan models
MaterialModels.ZhanGaussian.custom_pk1 = zhan_gaussian_pk1
MaterialModels.ZhanNonGaussian.custom_pk1 = zhan_nongaussian_pk1

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
