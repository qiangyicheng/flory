"""
Package :mod:`flory` provides methods and classes for analyzing phase separation in
multicomponent mixtures based on Flory-Huggins theory.
"""

try:
    # try reading version of the automatically generated module
    from ._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("flory")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"
    del PackageNotFoundError, version

from .interaction import FloryHugginsInteraction
from .entropy import IdealGasEntropy
from .ensemble import CanonicalEnsemble
from .free_energy import FloryHuggins
from .mcmp import CoexistingPhasesFinder
from .utils import find_coexisting_phases
