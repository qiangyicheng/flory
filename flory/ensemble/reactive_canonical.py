"""Module for reactive canonical ensemble of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np
from numba import float64, int32
from numba.experimental import jitclass

from .base import EnsembleBase, EnsembleBaseCompiled


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_weights", float64[:, :]),
        ("_weights_inv", float64[:, :]),
        ("_targets", float64[::1]),  # a C-continuous array
    ]
)
class ReactiveCanonicalEnsembleCompiled(EnsembleBaseCompiled):
    r"""Compiled class for canonical ensemble with conserved combinations of fractions.

    In contrast to the canonical ensemble, where the average volume fractions of the
    components are fixed, we here only fix a linear combination of the fractions, which
    correspond to conserved quantities :math:`\bar\psi_\beta` of reactions,

        .. math::
            \bar\psi_\beta = \sum_i B_{\beta,i} \bar\phi_i
            = \frac{\sum_m J_m \sum_i B_{\beta,i} \phi_i^{(m)} }{\sum_m J_m}

    Idea: This constraint could be implemented by guessing `phi_means` and using a
    projection method to enforce the linear constraint.
    """

    def __init__(
        self, weights: np.ndarray, targets: np.ndarray, *, weights_inv: np.ndarray
    ):
        r"""
        Args:
            weights:
                The weights :math:`B_{\beta,i}` with which each component contributes to
                the constraints.
            targets:
                The average value of the constraint :math:`\bar\psi_\beta`.
            weights_inv:
                The pseudo-inverse of `weights`
        """
        self._num_comp = weights.shape[1]
        self._weights = weights
        self._weights_inv = weights_inv
        self._targets = targets

    @property
    def num_comp(self):
        return self._num_comp

    def normalize(
        self, phis_comp: np.ndarray, Qs: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        r"""Normalize component fractions in canonical ensemble.

        Args:
            phis_comp:
                Mutable component Boltzmann factors, updated in-place to normalized
                fractions.
            Qs:
                Single molecule partition functions of components.
            masks:
                Masks indicating whether compartments are active.

        Returns:
            The incompressibility in each compartment.
        """
        # guess average fractions according to partition functions
        phi_means = Qs / Qs.sum()
        # project average fractions, so they obey constraints
        phi_means -= self._weights_inv @ (self._weights @ phi_means - self._targets)
        # normalize phi_means
        phi_means /= phi_means.sum()

        # enforce individual average fractions
        incomp = -1.0 * np.ones_like(phis_comp[0])
        for itr_comp in range(self._num_comp):
            factor = phi_means[itr_comp] / Qs[itr_comp]
            # in place update all compartments; replaces the Boltzmann factors stored in
            # phis_comp by the updated estimate of the volume fractions
            phis_comp[itr_comp] = factor * phis_comp[itr_comp] * masks
            incomp += phis_comp[itr_comp]
        incomp *= masks
        return incomp


class ReactiveCanonicalEnsemble(EnsembleBase):
    r"""Class for an canonical ensemble with conserved linear combinations of fractions.

    The particular form of the conservation law reads

    .. math::
            \bar\psi_\beta = \sum_i B_{\beta,i} \bar\phi_i
            = \frac{\sum_m J_m \sum_i B_{\beta,i} \phi_i^{(m)} }{\sum_m J_m}

    This reduces to the canonical ensemble for :math:`B_{\beta,i} = \delta_{\beta,i}`,
    where the average fractions are constrained, :math:`\bar\psi_\beta = \bar\phi_i`.
    In other cases, the constraints can capture chemical reactions, where only the total
    particles counts are conserved.
    """

    def __init__(self, num_comp: int, weights: np.ndarray, targets: np.ndarray):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            weights:
                The weights :math:`B_{\beta,i}` with which each component contributes to
                the constraints.
            targets:
                The average value of the constraint :math:`\bar\psi_\beta`.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)

        self._weights = np.asarray(weights, dtype=float)
        if self._weights.ndim != 2:
            raise ValueError("Weights must be 2d array")
        if self._weights.shape[1] != self.num_comp:
            raise ValueError("Second dimension of `weights` must equal component count")
        self.targets = targets

    @property
    def weights(self) -> np.ndarray:
        r"""The weights :math:`B_{\beta,i}` for each component."""
        return self._weights

    @property
    def targets(self) -> np.ndarray:
        r"""The average value of the constraint :math:`\bar\psi_\beta`."""
        return self._targets

    @targets.setter
    def targets(self, targets_new: np.ndarray):
        r"""Set the average values of the constraints.

        Args:
            targets_new:
                Updated average value of the constraint :math:`\bar\psi_\beta`.
        """
        self._targets = np.array(targets_new, dtype=float, copy=True)  # copy data

    def _compiled_impl(self) -> ReactiveCanonicalEnsembleCompiled:
        """Implementation of creating a compiled ensemble instance.

        This method overwrites the interface
        :meth:`~flory.ensemble.base.EnsembleBase._compiled_impl` in
        :class:`~flory.ensemble.base.EnsembleBase`.

        Returns:
            : Instance of :class:`CanonicalEnsembleCompiled`.
        """
        return ReactiveCanonicalEnsembleCompiled(
            self.weights, self.targets, weights_inv=np.linalg.pinv(self.weights)
        )
