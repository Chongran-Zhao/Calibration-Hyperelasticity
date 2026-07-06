"""Generalized strain measures E(lambda) for Hill-class models."""

from __future__ import annotations

import sympy as sp


class GeneralizedStrains:
    """Library of generalized strain measures ``E(lambda)``."""

    @staticmethod
    def SethHill(lambda_, params):
        """E = (1/m) * (lambda^m - 1)"""
        m = params["m"]
        return (1 / m) * (lambda_**m - 1)

    @staticmethod
    def Hencky(lambda_, params=None):
        """E = ln(lambda)"""
        return sp.log(lambda_)

    @staticmethod
    def CurnierRakotomanana(lambda_, params):
        """E = (1 / (m + n)) * (lambda^m - lambda^-n)"""
        m = params["m"]
        n = params["n"]
        return (1 / (m + n)) * (lambda_**m - lambda_**(-n))

    @staticmethod
    def CurnierZysset(lambda_, params):
        """E = ((2+m)/8)*lambda^2 - ((2-m)/8)*lambda^-2 - m/4"""
        m = params["m"]
        return ((2 + m) / 8) * lambda_**2 - ((2 - m) / 8) * lambda_**(-2) - (m / 4)

    @staticmethod
    def DarijaniNaghdabadi(lambda_, params):
        """E = (1 / (m + n)) * (exp(m*(lambda-1)) - exp(n*(lambda^-1 - 1)))"""
        m = params["m"]
        n = params["n"]
        term1 = sp.exp(m * (lambda_ - 1))
        term2 = sp.exp(n * (lambda_**-1 - 1))
        return (1 / (m + n)) * (term1 - term2)


def strain_function(strain_name):
    """Resolve a display name like ``"Seth-Hill"`` to its method."""
    try:
        return getattr(GeneralizedStrains, strain_name.replace("-", ""))
    except AttributeError:
        raise ValueError(f"Unknown strain measure: {strain_name}") from None


# Parameter metadata per strain measure. Keys are display names; the matching
# method is the key with hyphens stripped (see strain_function).
STRAIN_CONFIGS = {
    "Seth-Hill": {
        "params": ["m"],
        "defaults": [2.0],
        # Prevent the m=0 singularity and overflow; this range covers polymers.
        "bounds": [(0.1, 20.0)],
    },
    "Hencky": {
        "params": [],
        "defaults": [],
        "bounds": [],
    },
    "Curnier-Rakotomanana": {
        "params": ["m", "n"],
        "defaults": [1.0, 1.0],
        # Lower bound 0.01 keeps m+n away from zero.
        "bounds": [(0.01, 20.0), (0.01, 20.0)],
    },
    "Curnier-Zysset": {
        "params": ["m"],
        "defaults": [1.0],
        "bounds": [(-2.0, 2.0)],  # validity range of the measure
    },
    "Darijani-Naghdabadi": {
        "params": ["m", "n"],
        "defaults": [1.0, 1.0],
        "bounds": [(0.01, 20.0), (0.01, 20.0)],
    },
}

STRAIN_FORMULAS = {
    "Seth-Hill": r"E(\lambda) = \frac{1}{m}\left(\lambda^m - 1\right)",
    "Hencky": r"E(\lambda) = \ln(\lambda)",
    "Curnier-Rakotomanana": r"E(\lambda) = \frac{1}{m+n}\left(\lambda^m - \lambda^{-n}\right)",
    "Curnier-Zysset": r"E(\lambda) = \frac{2+m}{8}\lambda^2 - \frac{2-m}{8}\lambda^{-2} - \frac{m}{4}",
    "Darijani-Naghdabadi": r"E(\lambda) = \frac{1}{m+n}\left(e^{m(\lambda-1)} - e^{n(\lambda^{-1}-1)}\right)",
}
