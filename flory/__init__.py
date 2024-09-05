"""Package for finding coexisting phases of multicomponent mixtures.

The package :mod:`flory` provides methods and classes for analyzing phase separation in
general multicomponent mixtures. The package is inspired by the widely-used Flory-Huggins
model, but not limited to it. Instead, it describes general free energy using four
parts: :mod:`~flory.interaction`, :mod:`~flory.entropy`, :mod:`~flory.ensemble`, and
:mod:`~flory.constraint`. Coexisting phases are obtained using an improved Gibbs
ensemble method.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

try:
    # try reading version of the automatically generated module
    from ._version import __version__  # type: ignore
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("flory")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"
    del PackageNotFoundError, version

from .common import Phases
from .constraint import (
    ConstraintBase,
    LinearGlobalConstraint,
    LinearLocalConstraint,
    NoConstraint,
)
from .ensemble import CanonicalEnsemble, EnsembleBase, GrandCanonicalEnsemble
from .entropy import EntropyBase, IdealGasEntropy
from .free_energy import FloryHuggins, FreeEnergyBase
from .interaction import FloryHugginsInteraction, InteractionBase
from .mcmp import CoexistingPhasesFinder
from .shortcut import find_coexisting_phases
