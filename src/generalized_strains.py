import sympy as sp

class GeneralizedStrains:
    """
    Library of generalized strain measures E(lambda).
    """

    @staticmethod
    def SethHill(lambda_, params):
        """E = (1/m) * (lambda^m - 1)"""
        m = params['m']
        # Handle limit case m->0 (Logarithmic) if needed, but usually handled by optimizer not hitting exactly 0
        return (1 / m) * (lambda_**m - 1)

    @staticmethod
    def Hencky(lambda_, params=None):
        """E = ln(lambda)"""
        return sp.log(lambda_)

    @staticmethod
    def CurnierRakotomanana(lambda_, params):
        """E = (1 / (m + n)) * (lambda^m - lambda^-n)"""
        m = params['m']
        n = params['n']
        return (1 / (m + n)) * (lambda_**m - lambda_**(-n))

    @staticmethod
    def CurnierZysset(lambda_, params):
        """E = ((2+m)/8)*lambda^2 - ((2-m)/8)*lambda^-2 - m/4"""
        m = params['m']
        return ((2 + m) / 8) * lambda_**2 - ((2 - m) / 8) * lambda_**(-2) - (m / 4)

    @staticmethod
    def DarijaniNaghdabadi(lambda_, params):
        """E = (1 / (m + n)) * ( exp(m*(lambda-1)) - exp(n*(lambda^-1 - 1)) )"""
        m = params['m']
        n = params['n']
        term1 = sp.exp(m * (lambda_ - 1))
        term2 = sp.exp(n * (lambda_**-1 - 1))
        return (1 / (m + n)) * (term1 - term2)

# =============================================================================
# Generalized Strain Configuration (Parameter Metadata)
# =============================================================================
# Keys are display names. Since Python method names cannot contain hyphens,
# logic in material_models.py will strip hyphens ('-') to find the matching method.
STRAIN_CONFIGS = {
    "Seth-Hill": {
        "params": ["m"],
        "defaults": [2.0],          
        # Prevent m=0 singularity and overflow. 
        # Range [-20, 20] is physically sufficient for most polymers.
        "bounds": [(0.1, 20.0)]     
    },
    "Hencky": {
        "params": [],               
        "defaults": [],
        "bounds": []
    },
    "Curnier-Rakotomanana": {
        "params": ["m", "n"],
        "defaults": [1.0, 1.0],     
        # Prevent m+n=0 singularity and overflow.
        # Changed lower bound from 0.0 to 0.01 to avoid division by zero.
        "bounds": [(0.01, 20.0), (0.01, 20.0)] 
    },
    "Curnier-Zysset": {
        "params": ["m"],
        "defaults": [1.0],
        "bounds": [(-2.0, 2.0)]     # Validity range for this model
    },
    "Darijani-Naghdabadi": {
        "params": ["m", "n"],
        "defaults": [1.0, 1.0],
        # Prevent m+n=0 and overflow
        "bounds": [(0.01, 20.0), (0.01, 20.0)] 
    }
}