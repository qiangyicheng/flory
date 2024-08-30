"""Module for linear global constraint.

"""
from __future__ import annotations

import logging

import numpy as np
from numba import float64, int32
from numba.experimental import jitclass

from .base import ConstraintBase, ConstraintBaseCompiled


@jitclass(
    [
        ("_num_cons", int32),  # a scalar
        ("_num_feat", int32),  # a scalar
        ("_Cs", float64[:, ::1]),  # a C-continuous array
        ("_Ts", float64[::1]),  # a C-continuous array
        ("_multiplier", float64[::1]),  # a C-continuous array
        ("_residue", float64[::1]),  # a C-continuous array
        ("_potential", float64[:, ::1]),  # a C-continuous array
        ("_volume_derivative", float64[::1]),  # a C-continuous array
        ("_acceptance_ratio", float64),  # a C-continuous array
        ("_elasticity", float64),  # a C-continuous array
    ]
)
class LinearGlobalConstraintCompiled(ConstraintBaseCompiled):
    r"""Compiled class for linear global constraint.

    Linear global constraint requires that the certain linear combination of feature
    average volume fractions :math:`\bar{\phi}_r` are constant,

        .. math::
            \sum_r C_{\alpha,r} \sum_m J_m \phi_r^{(m)} = T_\alpha \sum_m J_m,

    where :math:`\alpha` is the index of constraint. This effectively means an additional
    term in the free energy,

        .. math::
            f_\mathrm{constraint} = \sum_\alpha^A \xi_\alpha \sum_m^{M} J_m \left(\sum_r C_{\alpha,r} \phi_r^{(m)} - T_\alpha\right).

    However, such form usually suffers from numerical instability since the Lagrange
    multiplier only delivers good guidance when the constraint is almost satisfied.
    We thus extend the term further into,

        .. math::
            f_\mathrm{constraint} = \sum_\alpha^A \xi_\alpha \sum_m^{M} J_m \left(\sum_r C_{\alpha,r} \phi_r^{(m)} - T_\alpha\right) +
            \sum_\alpha^A \kappa \left[ \sum_m^{M} J_m \left(\sum_r C_{\alpha,r} \phi_r^{(m)} - T_\alpha\right) \right]^2

    where we term :math:`\kappa` as the elasticity of constraints. Note that when the
    constraints are satisfied, these additional terms vanish.
    """

    def __init__(
        self, Cs: np.ndarray, Ts: np.ndarray, acceptance_ratio: float, elasticity: float
    ):
        r"""
        Args:
            Cs:
                2D array with the size of :math:`A \times N_\mathrm{s}`,
                containing coefficients of features for linear constraints. Note that both
                number of features :math:`N_\mathrm{s}` and number of constraints
                :math:`A` are inferred from this parameter.
            Ts:
                1D vector with the size of :math:`A`, containing the
                targets of the constraints.
            acceptance_ratio:
                The relative acceptance during :meth:`evolve`.
            elasticity:
                The additional elastic constant to guide when the Lagrange multiplier is
                inefficient.
        """
        self._num_cons = Cs.shape[0]
        self._num_feat = Cs.shape[1]
        self._Cs = Cs
        self._Ts = Ts
        self._acceptance_ratio = acceptance_ratio
        self._elasticity = elasticity
        self._multiplier = np.zeros(self._num_cons)
        self._potential = np.zeros((self._num_feat, 1))
        self._residue = np.zeros(self._num_cons)
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
        self._multiplier = np.zeros(self._num_cons)

    def prepare(self, phis_feat: np.ndarray, Js: np.ndarray, masks: np.ndarray) -> None:
        compart_residue = self._Cs @ phis_feat
        for itr_cons in range(self._num_cons):
            compart_residue[itr_cons] -= self._Ts[itr_cons]

        self._residue = (compart_residue * Js).sum(axis=-1)

        self._potential = np.outer(
            self._Cs.T @ (self._multiplier + 2.0 * self._elasticity * self._residue),
            np.ones_like(phis_feat[0]),
        )

        self._volume_derivative = np.zeros_like(phis_feat[0])
        for itr_cons in range(self._num_cons):
            self._volume_derivative += self._residue[itr_cons] * (
                self._multiplier[itr_cons]
                + 2.0 * compart_residue[itr_cons] * self._elasticity
            )
        self._potential *= masks
        self._residue *= masks
        self._volume_derivative *= masks

        self._residue /= Js.sum() # scale residue according to total volume

    def evolve(self, step: float, masks: np.ndarray) -> float:
        self._multiplier += step * self._acceptance_ratio * self._residue
        self._multiplier *= masks
        return np.abs(self._residue).max()


class LinearGlobalConstraint(ConstraintBase):
    r"""Class for for linear global constraints.

    The linear global constraints require that

        .. math::
            \sum_r C_{\alpha,r} \sum_m J_m \phi_r^{(m)} = T_\alpha \sum_m J_m,

    where :math:`\alpha` is the index of constraint.
    """

    def __init__(self, num_feat: int, Cs: np.ndarray, Ts: np.ndarray):
        r"""
        Args:
            num_feat:
                Number of features :math:`N_\mathrm{s}`.
            Cs:
                2D array with the size of :math:`A \times N_\mathrm{s}`,
                containing coefficients of features for linear constraints. Note that both
                number of features :math:`N_\mathrm{s}` and number of constraints
                :math:`A` are inferred from this parameter.
            Ts:
                1D vector with the size of :math:`A`, containing the
                targets of the constraints.
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
        self, constraint_acceptance_ratio: float = 1.0, constraint_elasticity: float = 1.0
    ) -> object:
        r"""Implementation of creating a compiled constraint instance.

        This method overwrites the interface
        :meth:`~flory.constraint.base.ConstraintBase._compiled_impl` in
        :class:`~flory.constraint.base.ConstraintBase`.

        Args:
            constraint_acceptance_ratio:
                Relative acceptance for the evolution of the Lagrange multipliers of the
                constraints. A value of 1 indicates the multipliers are evolved in the
                same pace as the conjugate fields :math:`w_r^{(m)}`.
            constraint_elasticity:
                Elasticity :math:`\kappa` of the constraints.

        Returns:
            : Instance of :class:`LinearGlobalConstraintCompiled`.
        """
        return LinearGlobalConstraintCompiled(
            self.Cs, self.Ts, constraint_acceptance_ratio, constraint_elasticity
        )
