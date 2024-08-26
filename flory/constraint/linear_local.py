"""Module for linear local constraint.

"""

from typing import Optional, Tuple
import logging

from numba.experimental import jitclass
from numba import float64, int32
import numpy as np
from .base import ConstraintBase, ConstraintBaseCompiled


@jitclass(
    [
        ("_num_cons", int32),  # a scalar
        ("_num_feat", int32),  # a scalar
        ("_Cs", float64[:, ::1]),  # a C-continuous array
        ("_Ts", float64[::1]),  # a C-continuous array
        ("_multiplier", float64[:, ::1]),  # a C-continuous array
        ("_residue", float64[:, ::1]),  # a C-continuous array
        ("_field", float64[:, ::1]),  # a C-continuous array
        ("_volume_derivative", float64[::1]),  # a C-continuous array
        ("_acceptance_ratio", float64),  # a C-continuous array
        ("_elasticity", float64),  # a C-continuous array
    ]
)
class LinearLocalConstraintCompiled(ConstraintBaseCompiled):
    r"""Compiled class for linear local constraint.
    Linear local constraint requires that the certain linear combination of feature volume fractions are constant,

        .. math::
            \sum_i C_i^s \phi_i^{(\alpha)} &= T_s,

    where :math:`s` is the index of constraint.
    """

    def __init__(
        self, Cs: np.ndarray, Ts: np.ndarray, acceptance_ratio: float, elasticity: float
    ):
        r"""
        Args:
        """
        self._num_cons = Cs.shape[0]
        self._num_feat = Cs.shape[1]
        self._Cs = Cs
        self._Ts = Ts
        self._acceptance_ratio = acceptance_ratio
        self._elasticity = elasticity
        self._multiplier = np.zeros((self._num_cons, 1))
        self._field = np.zeros((self._num_feat, 1))
        self._residue = np.zeros((self._num_cons, 1))
        self._volume_derivative = np.zeros((1,))

    @property
    def num_feat(self):
        r"""Number of features :math:`N_\mathrm{f}`."""
        return self._num_feat

    @property
    def field(self):
        return self._field

    @property
    def volume_derivative(self):
        return self._volume_derivative

    def initialize(self, num_part: int):
        self._multiplier = np.zeros((self._num_cons, num_part))

    def prepare(self, phis_feat: np.ndarray, masks: np.ndarray):
        self._residue = self._Cs @ phis_feat
        for itr_cons in range(self._num_cons):
            self._residue[itr_cons] -= self._Ts[itr_cons]

        self._field = self._Cs.T @ (
            self._multiplier + 2 * self._elasticity * self._residue
        )
        self._volume_derivative = np.zeros_like(self._residue[0])
        for itr_cons in range(self._num_cons):
            self._volume_derivative += self._residue[itr_cons] * (
                self._multiplier[itr_cons] + self._residue[itr_cons] * self._elasticity
            )
        self._field *= masks
        self._residue *= masks
        self._volume_derivative *= masks

    def evolve(self, step: float, masks: np.ndarray) -> float:
        self._multiplier += step * self._acceptance_ratio * self._residue
        self._multiplier *= masks
        return np.abs(self._residue).max()


class LinearLocalConstraint(ConstraintBase):
    r""" """

    def __init__(self, num_feat: int, Cs: np.ndarray, Ts: np.ndarray):
        r"""
        Args:
            num_feat:
                Number of features :math:`N_\mathrm{f}`.
        """
        super().__init__(num_feat)
        self._logger = logging.getLogger(self.__class__.__name__)

        Cs = np.atleast_1d(Cs)
        if Cs.ndim == 1:
            self.num_cons = 1
        elif Cs.ndim == 2:
            self.num_cons = Cs.shape[0]
        else:
            self._logger("Constraint matrix is not 1D or 2D.")
            raise ValueError("Constraint matrix must be 1D or 2D.")

        shape = (self.num_cons, self.num_feat)
        self.Cs = np.broadcast_to(Cs, shape).astype(float)
        shape = (self.num_cons,)
        Ts = np.atleast_1d(Ts)
        self.Ts = np.broadcast_to(Ts, shape).astype(float)

    def _compiled_impl(
        self, constraint_acceptance: float = 1.0, constraint_elasticity: float = 1.0
    ) -> object:
        """Implementation of creating a compiled constraint instance.
        This function overwrites the interface
        :meth:`~flory.ensemble.base.ConstraintBase._compiled_impl` in
        :class:`~flory.ensemble.base.ConstraintBase`.

        Returns:
            Instance of :class:`LinearLocalConstraintCompiled`.
        """
        return LinearLocalConstraintCompiled(
            self.Cs, self.Ts, constraint_acceptance, constraint_elasticity
        )
