"""Exceptions that package :mod:`flory` might raise.
"""

class VolumeFractionError(ValueError):
    """Error indicating that the volume fraction is smaller than 0."""

    pass

class ComponentNumberError(ValueError):
    """Error indicating mismatch of number of components."""

    pass

class FeatureNumberError(ValueError):
    """Error indicating mismatch of number of features."""

    pass

