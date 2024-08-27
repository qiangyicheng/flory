"""Module for no constraint.

"""

import logging

from numba.experimental import jitclass
from numba import float64, int32
import numpy as np
from .base import ConstraintBase, ConstraintBaseCompiled


@jitclass(
    [
        ("_num_feat", int32),  # a scalar
        ("_potential", float64[:, ::1]),  # a C-continuous array
        ("_volume_derivative", float64[::1]),  # a C-continuous array
    ]
)
class NoConstraintCompiled(ConstraintBaseCompiled):
    r"""Compiled class for no constraint."""

    def __init__(self, num_feat):
        r"""
        Args:
            num_feat:
                Number of features :math:`N_\mathrm{s}`.
        """
        self._num_feat = num_feat
        self._potential = np.zeros((self._num_feat, 1))
        self._volume_derivative = np.zeros((1,))

    @property
    def num_feat(self) -> int:
        return self._num_feat

    @property
    def potential(self) -> np.ndarray:
        return self._potential

    @property
    def volume_derivative(self) -> np.ndarray:
        return self._volume_derivative

    def initialize(self, num_part: int) -> None:
        self._potential = np.zeros((self._num_feat, num_part))
        self._volume_derivative = np.zeros((num_part,))
        pass

    def prepare(self, phis_feat: np.ndarray, masks: np.ndarray) -> None:
        pass

    def evolve(self, step: float, masks: np.ndarray) -> float:
        return 0

# No need for a factory class