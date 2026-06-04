"""Module for canonical ensemble of mixture.

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
        ("_weights", float64[::1, ::1]),
        ("_means", float64[::1]),  # a C-continuous array
    ]
)
class CanonicalReactionsEnsembleCompiled(EnsembleBaseCompiled):
    r"""Compiled class for canonical ensemble with conserved combinations of fractions.

    In contrast to the canonical ensemble, where the average volume fractions of the
    components are fixed, we here only fix a linear combination of the fractions, which
    correspond to conserved quantities :math:`\bar\psi_\beta` of reactions,

        .. math::
            \bar\psi_\beta = \frac{\sum_m J_m \sum_i B_{\beta,i} \phi_i^{(m)} }{\sum_m J_m}

    Idea: This constraint could be implemented using a projection method if we have good
    estimates of the current average fraction

    Therefore, the volume fractions distribution of the components in compartments can be
    obtained by normalizing the Boltzmann factors according to the average volume
    fractions,

        .. math::
            \phi_i^{(m)} &= \frac{\bar{\phi}_i}{Q_i} p_i^{(m)} \\
            Q_i &= \sum_m p_i^{(m)} J_m .

    Since (translational) entropy is always defined for each component, this class is only
    aware of the component-based description of the system.
    """

    def __init__(self, weights: np.ndarray, means: np.ndarray):
        r"""
        Args:
            phi_means:
                1D array with the size of :math:`N_\mathrm{C}`, containing the mean volume
                fractions of the components, :math:`\bar{\phi}_i`. The number of
                components :math:`N_\mathrm{C}` is inferred from this array.
        """
        self._num_comp = means.shape[0]
        self._weights = weights
        self._means = means

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
        # determine average fractions according to constraints

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


class CanonicalReactionsEnsemble(EnsembleBase):
    r"""Class for an canonical ensemble with conserved linear combinations of fractions.

    The particular form of the conservation law reads

    .. math::
        \bar\psi_\beta = \frac{\sum_m J_m \sum_i B_{\beta,i} \phi_i^{(m)} }{\sum_m J_m}.

    This reduces to the canonical ensemble for :math:`B_{\beta,i} = \delta_{r,i}` and the
    constraints are simply the average fractions, :math:`\bar\psi_\beta = \bar\phi_i`.
    """

    def __init__(self, num_comp: int, weights: np.ndarray, means: np.ndarray):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            weights:
                The weights :math:`B_{\beta,i}` with which each component contributes to
                the constraints.
            means:
                The average value of the constraint :math:`\bar\psi_\beta`.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)

        self._weights = np.atleast_2d(weights)
        if self._weights.shape[1] != self.num_comp:
            raise ValueError(
                "Second dimension of `weights` must equal component count."
            )
        self.means = means

    @property
    def weights(self) -> np.ndarray:
        r"""The weights :math:`B_{\beta,i}` for each component."""
        return self._weights

    @property
    def means(self) -> np.ndarray:
        r"""The average value of the constraint :math:`\bar\psi_\beta`."""
        return self._means

    @means.setter
    def means(self, means_new: np.ndarray):
        r"""Set the average values of the constraints.

        Args:
            means_new:
                Updated average value of the constraint :math:`\bar\psi_\beta`.
        """
        means_new = np.array(means_new)  # copy data
        self._means = np.broadcast_to(means_new, (self.num_comp,))

        # if not np.isclose(self._means.sum(), 1.0):
        #     self._logger.warning(
        #         "The sum of phi_means is not 1. In incompressible system the iteration may never converge."
        #     )

    def _compiled_impl(self) -> CanonicalReactionsEnsembleCompiled:
        """Implementation of creating a compiled ensemble instance.

        This method overwrites the interface
        :meth:`~flory.ensemble.base.EnsembleBase._compiled_impl` in
        :class:`~flory.ensemble.base.EnsembleBase`.

        Returns:
            : Instance of :class:`CanonicalEnsembleCompiled`.
        """
        return CanonicalReactionsEnsembleCompiled(self.weights, self.means)
