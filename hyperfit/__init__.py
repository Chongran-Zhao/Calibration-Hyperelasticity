"""hyperfit: calibration of hyperelastic material models.

Core public API:

* :class:`MaterialModels` -- model registry and factories
* :class:`Kinematics` -- stress evaluation for a tagged model
* :class:`MaterialOptimizer` -- parameter calibration
* :class:`ParallelNetwork` -- parallel composition of models
* :func:`load_experimental_data` / :func:`load_experimental_data_h5`
* :func:`plot_comparison`
"""

from .datasets import DataLoadError, load_experimental_data, load_experimental_data_h5
from .kinematics import Kinematics
from .mechanics import get_deformation_gradient, get_stress_components, inv_Langevin_Kroger
from .models import MaterialModels, register_model
from .network import ParallelNetwork
from .optimizer import MaterialOptimizer, OptimizationAbortedError
from .strains import GeneralizedStrains, STRAIN_CONFIGS, STRAIN_FORMULAS

__version__ = "0.2.0"

__all__ = [
    "DataLoadError",
    "GeneralizedStrains",
    "Kinematics",
    "MaterialModels",
    "MaterialOptimizer",
    "OptimizationAbortedError",
    "ParallelNetwork",
    "STRAIN_CONFIGS",
    "STRAIN_FORMULAS",
    "get_deformation_gradient",
    "get_stress_components",
    "inv_Langevin_Kroger",
    "load_experimental_data",
    "load_experimental_data_h5",
    "register_model",
    "__version__",
]


def plot_comparison(*args, **kwargs):
    """Lazy proxy so importing hyperfit does not require matplotlib."""
    from .plotting import plot_comparison as _plot_comparison

    return _plot_comparison(*args, **kwargs)
