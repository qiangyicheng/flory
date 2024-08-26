"""Module for canonical ensemble of mixture.

"""

from typing import Optional, Tuple
import logging

from numba.experimental import jitclass
from numba import float64, int32
import numpy as np
from .base import EnsembleBase, EnsembleBaseCompiled


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_phi_means", float64[::1]),  # a C-continuous array
    ]
)
class CanonicalEnsembleCompiled(EnsembleBaseCompiled):
    r"""Compiled class for canonical ensemble.
    In canonical ensemble, the average volume fractions of the components are fixed.
    Therefore, the volume fractions distribution of the components in compartments can be
    obtained by normalizing the Boltzmann factors according to the average volume
    fractions,
    
        .. math::
            \phi_i^{(m)} &= \frac{\bar{\phi}_i}{Q_i} p_i^{(m)} \\
            Q_i &= \sum_m p_i^{(m)} J_m .
    
    Since (translational) entropy is always defined for each component, this class is only
    aware of the component-based description of the system.
    """
    def __init__(self, phi_means: np.ndarray):
        r"""
        Args:
            phi_means:
                The 1D array with the size of :math:`N_\mathrm{c}`, containing the mean
                volume fractions of the components :math:`i`, :math:`\bar{\phi}_i`. The
                number of components :math:`N_\mathrm{c}` is inferred from this matrix.
        """
        self._num_comp = phi_means.shape[0]
        self._phi_means = phi_means  # do not affect chis

    @property
    def num_comp(self):
        return self._num_comp

    def normalize(
        self, phis_comp: np.ndarray, Qs: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        incomp = -1.0 * np.ones_like(phis_comp[0])
        for itr_comp in range(self._num_comp):
            factor = self.phi_means[itr_comp] / Qs[itr_comp]
            phis_comp[itr_comp] = factor * phis_comp[itr_comp] * masks
            incomp += phis_comp[itr_comp]
        incomp *= masks
        return incomp
   


class CanonicalEnsemble(EnsembleBase):
    r"""Class for an canonical ensemble that the volume fractions are conserved,
    
    .. math::
        \bar{\phi}_i = \frac{\sum_m \phi_i^{(m)} J_m }{\sum_m J_m}.

    """

    def __init__(
        self,
        num_comp: int,
        phi_means: np.ndarray,
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`.
            phi_means:
                The mean volume fractions of the components :math:`\bar{\phi}_i`.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)

        phi_means = np.atleast_1d(phi_means)

        shape = (num_comp,)
        self.phi_means = np.array(np.broadcast_to(phi_means, shape))

        if not np.allclose(self.phi_means.sum(), 1.0):
             self._logger.warning(
                "The sum of phi_means is not 1. In incompressible system the iteration may never converge."
            )

    def _compiled_impl(self) -> object:
        """Implementation of creating a compiled ensemble instance.
        This function overwrites the interface
        :meth:`~flory.ensemble.base.EnsembleBase._compiled_impl` in
        :class:`~flory.ensemble.base.EnsembleBase`. 
        
        Returns:
            : Instance of :class:`CanonicalEnsembleCompiled`.
        """

        return CanonicalEnsembleCompiled(self.phi_means)
