import sympy as sp

class GeneralizedStrains:
    """
    Library of generalized strain measures E(lambda) based on the provided literature.
    
    Families included:
    1. Seth-Hill
    2. Hencky
    3. Curnier-Rakotomanana
    4. Curnier-Zysset
    5. Darijani-Naghdabadi
    """

    @staticmethod
    def SethHill(lambda_, params):
        """
        Seth-Hill generalized strain.
        E = (1/m) * (lambda^m - 1)
        
        Args:
            params: {'m': symbol} (m != 0)
        """
        m = params['m']
        return (1 / m) * (lambda_**m - 1)

    @staticmethod
    def Hencky(lambda_, params=None):
        """
        Hencky (Logarithmic) strain.
        E = ln(lambda)
        """
        return sp.log(lambda_)

    @staticmethod
    def CurnierRakotomanana(lambda_, params):
        """
        Curnier-Rakotomanana generalized strain.
        E = (1 / (m + n)) * (lambda^m - lambda^-n)
        
        Args:
            params: {'m': symbol, 'n': symbol} (mn > 0)
        """
        m = params['m']
        n = params['n']
        return (1 / (m + n)) * (lambda_**m - lambda_**(-n))

    @staticmethod
    def CurnierZysset(lambda_, params):
        """
        Curnier-Zysset generalized strain.
        E = ((2+m)/8)*lambda^2 - ((2-m)/8)*lambda^-2 - m/4
        
        Args:
            params: {'m': symbol} (-2 <= m <= 2)
        """
        m = params['m']
        return ((2 + m) / 8) * lambda_**2 - ((2 - m) / 8) * lambda_**(-2) - (m / 4)

    @staticmethod
    def DarijaniNaghdabadi(lambda_, params):
        """
        Darijani-Naghdabadi generalized strain.
        E = (1 / (m + n)) * ( exp(m*(lambda-1)) - exp(n*(lambda^-1 - 1)) )
        
        Args:
            params: {'m': symbol, 'n': symbol} (m, n > 0)
        """
        m = params['m']
        n = params['n']
        term1 = sp.exp(m * (lambda_ - 1))
        term2 = sp.exp(n * (lambda_**-1 - 1))
        return (1 / (m + n)) * (term1 - term2)
# EOF
