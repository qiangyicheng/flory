"""
Package :mod:`flory` provides methods and classes for phase separation in multicomponent mixtures based on Flory-Huggins theory.
"""

try:
    # try reading version of the automatically generated module
    from ._version import __version__
except ImportError:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("flory")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"
    del PackageNotFoundError, version

from .mcmp import CoexistingPhasesFinder, find_coexisting_phases