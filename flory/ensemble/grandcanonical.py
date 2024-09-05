"""Module for grand canonical ensemble of mixture.

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
        ("_scaled_activity", float64[::1]),  # a C-continuous array
    ]
)
class GrandCanonicalEnsembleCompiled(EnsembleBaseCompiled):
    r"""Compiled class for grand canonical ensemble.

    In grand canonical ensemble, the original chemical potentials of the components are fixed.
    Therefore, the volume fractions distribution of the components in compartments can be
    obtained by scaling the Boltzmann factors according to the scaled activity,
    
        .. math::
            \phi_i^{(m)} &= l_i e^{l_i \mu_i} p_i^{(m)} \\
                
    where :math:`l_i e^{l_i \mu_i}` is the scaled activity, :math:`l_i` is the relative
    volumes of molecules and :math:`\mu_i` is the chemical potentials of the components by
    volume. Since (translational) entropy is always defined for each component, this class
    is only aware of the component-based description of the system.
    """

    def __init__(self, scaled_activity: np.ndarray):
        r"""
        Args:
            scaled_activity:
                1D array with the size of :math:`N_\mathrm{C}`, containing the scaled
                activities of the components, :math:`l_i e^{l_i \mu_i}`. The number
                of components :math:`N_\mathrm{C}` is inferred from this array.
        """
        self._num_comp = scaled_activity.shape[0]
        self._scaled_activity = scaled_activity  # do not affect chis

    @property
    def num_comp(self):
        return self._num_comp

    def normalize(
        self, phis_comp: np.ndarray, Qs: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        incomp = -1.0 * np.ones_like(phis_comp[0])
        for itr_comp in range(self._num_comp):
            phis_comp[itr_comp] = (
                self._scaled_activity[itr_comp] * phis_comp[itr_comp] * masks
            )
            incomp += phis_comp[itr_comp]
        incomp *= masks
        return incomp


class GrandCanonicalEnsemble(EnsembleBase):
    r"""Class for an grand canonical ensemble that the chemical potentials are fixed."""

    def __init__(
        self,
        num_comp: int,
        scaled_activity: np.ndarray,
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            scaled_activity:
                The scaled activities of the components :math:`l_i e^{l_i \mu_i}`.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)

        scaled_activity = np.atleast_1d(scaled_activity)

        shape = (num_comp,)
        self._scaled_activity = np.array(np.broadcast_to(scaled_activity, shape))

    @property
    def scaled_activity(self) -> np.ndarray:
        r"""The scaled activities of the components :math:`l_i e^{l_i \mu_i}`."""
        return self._scaled_activity

    @scaled_activity.setter
    def scaled_activity(self, scaled_activity_new: np.ndarray):
        scaled_activity_new = np.atleast_1d(scaled_activity_new)
        shape = (self.num_comp,)
        self._scaled_activity = np.array(np.broadcast_to(scaled_activity_new, shape))

    @classmethod
    def from_chemical_potential(
        cls, num_comp: int, mus: np.ndarray, sizes: np.ndarray | None = None
    ):
        r"""Create grand canonical ensemble from chemical potentials by volume.

        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            mus:
                The chemical potentials by volume :math:`\mu_i`.
            sizes:
                The relative molecule volumes :math:`l_i = \nu_i/\nu` with respect to the
                volume of a reference molecule :math:`\nu`. It is treated as all-one
                vector by default.
        """
        mus = np.atleast_1d(mus)

        shape = (num_comp,)
        mus = np.array(np.broadcast_to(mus, shape))

        if sizes is None:
            sizes = np.ones_like(mus)
        else:
            sizes = np.atleast_1d(sizes)
            sizes = np.array(np.broadcast_to(sizes, shape))

        scaled_activity = sizes * np.exp(sizes * mus)
        return cls(num_comp, scaled_activity)

    def _compiled_impl(self) -> GrandCanonicalEnsembleCompiled:
        """Implementation of creating a compiled ensemble instance.

        This method overwrites the interface
        :meth:`~flory.ensemble.base.EnsembleBase._compiled_impl` in
        :class:`~flory.ensemble.base.EnsembleBase`.

        Returns:
            : Instance of :class:`GrandCanonicalEnsembleCompiled`.
        """

        return GrandCanonicalEnsembleCompiled(self._scaled_activity)
