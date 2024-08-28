"""Package for finding coexisting phases of multicomponent mixtures.
Package :mod:`flory` provides methods and classes for analyzing phase separation in
general multicomponent mixtures. The package is inspired by the widely-used Flory-Huggins
model, but not limited to it. Package :mod:`flory` describes a general free energy by four
parts, :mod:`~flory.interaction`, :mod:`~flory.entropy`, :mod:`~flory.ensemble` and
:mod:`~flory.constraint`, and use a improved Gibbs ensemble method to obtain the
coexisting phases.
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

from .interaction import InteractionBase, FloryHugginsInteraction
from .entropy import EntropyBase, IdealGasEntropy
from .ensemble import EnsembleBase, CanonicalEnsemble
from .free_energy import FreeEnergyBase, FloryHuggins
from .constraint import ConstraintBase, LinearLocalConstraint
from .mcmp import CoexistingPhasesFinder
from .shortcut import find_coexisting_phases
