from typing import Optional, Tuple
import logging

from numba.experimental import jitclass
from numba import float64, int32
import numpy as np
from .base import EnsembleBase


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_num_feat", int32),  # a scalar
        ("_phi_means", float64[::1]),  # a C-continuous array
    ]
)
class CanonicalEnsembleCompiled(object):
    def __init__(self, phi_means: np.ndarray):
        r"""The compiled ideal gas entropy.

        Args:
            phi_means:
                The 1D array with the size of :math:`N_\\mathrm{c}`, containing the mean
                volume fractions of the components :math:`\\bar{\\phi}_i`. The number of
                components is inferred from this matrix.
        """
        self._num_comp = phi_means.shape[0]
        self._num_feat = phi_means.shape[0]
        self._phi_means = phi_means  # do not affect chis

    @property
    def num_comp(self):
        return self._num_comp

    @property
    def num_feat(self):
        return self._num_feat

    @property
    def phi_means(self):
        return self._phi_means

    def normalize(
        self, phis_comp: np.ndarray, Qs: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Normalize the volume fractions of features.
        Note that this function should modify the :paramref:`phis_comp`.

        Args:
            phis_comp:
                The Boltzmann factors of the components, which are proportional to
                resulting volume fractions.
            Qs:
                The single molecule partition functions of the components

        Returns:
            : The incompressibility
        """
        incomp = -1.0 * np.ones_like(phis_comp[0])
        for itr_comp in range(self._num_comp):
            factor = self.phi_means[itr_comp] / Qs[itr_comp]
            phis_comp[itr_comp] = factor * phis_comp[itr_comp] * masks
            incomp += phis_comp[itr_comp]
        incomp *= masks
        return incomp
   


class CanonicalEnsemble(EnsembleBase):
    r"""represents an canonical ensemble that the volume fractions are conserved."""

    def __init__(
        self,
        num_comp: int,
        phi_means: np.ndarray,
    ):
        """
        Args:
            num_comp:
                Number of components in the system
            phi_means:
                The mean volume fractions of the components :math:`\\bar{\\phi}_i`.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)
        self.num_comp = num_comp

        phi_means = np.atleast_1d(phi_means)

        shape = (num_comp,)
        self.phi_means = np.array(np.broadcast_to(phi_means, shape))

        if not np.allclose(self.phi_means.sum(), 1.0):
             self._logger.warning(
                "The sum of phi_means is not 1. In incompressible system the iteration may never converge."
            )

    def _compiled_impl(self) -> object:
        """make the entropy instance that can be used by the :mod:`mcmp` module."""

        return CanonicalEnsembleCompiled(self.phi_means)
