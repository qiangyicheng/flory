"""Module for polydispersed ideal gas entropic energy of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np
from numba import float64, int32
from numba.experimental import jitclass

from ..common import *
from .base import EntropyBaseCompiled
from .ideal_gas import IdealGasEntropyBase


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_num_feat", int32),  # a scalar
        ("_num_comp_per_feat", int32[::1]),  # a C-continuous array
        ("_sizes", float64[::1]),  # a C-continuous array
    ]
)
class IdealGasPolydispersedEntropyCompiled(EntropyBaseCompiled):
    r"""Compiled class for the entropic energy of mixture of polydispersed ideal gas.

    For ideal gas, the Boltzmann factor :math:`p_i^{(m)}` is determined by the relative
    volumes of the molecules :math:`l_i = \nu_i/\nu` and the mean fields it feel,
    :math:`w_r^{(m)}`,

        .. math::
            p_i^{(m)} = \exp(-l_i w_r^{(m)}).

    Note that this class assumes that there's degeneracy of the components' features, in
    other words several components in a group can correspond to one feature, such that all
    components in this group share the same field. Therefore the number of features
    :math:`N_\mathrm{S}` is larger than the number of components :math:`N_\mathrm{C}`.
    """

    def __init__(self, sizes: np.ndarray, num_comp_per_feat: np.ndarray):
        r"""
        Args:
            sizes:
                1D array with the size of :math:`N_\mathrm{C}`, containing the relative
                molecule volumes :math:`l_i = \nu_i/\nu`. The number of components
                :math:`N_\mathrm{C}` is inferred from this array.
            num_comp_per_feat:
                1D array with the size of :math:`N_\mathrm{S}`, containing the number
                component of each feature. The number of features :math:`N_\mathrm{S}` is
                inferred from this array. The sum of this array must be the number of
                components :math:`N_\mathrm{C}`.
        """
        self._num_comp = sizes.shape[0]
        self._num_feat = num_comp_per_feat.shape[0]
        self._sizes = sizes
        self._num_comp_per_feat = num_comp_per_feat

    @property
    def num_comp(self) -> int:
        return self._num_comp

    @property
    def num_feat(self) -> int:
        return self._num_feat

    def partition(
        self, phis_comp: np.ndarray, omegas: np.ndarray, Js: np.ndarray
    ) -> np.ndarray:
        Qs = np.zeros((self._num_comp,))
        total_Js = Js.sum()

        itr_comp = 0
        for itr_feat in range(self._num_feat):
            for _ in range(self._num_comp_per_feat[itr_feat]):
                phis_comp[itr_comp] = np.exp(-omegas[itr_feat] * self._sizes[itr_comp])
                Qs[itr_comp] = (phis_comp[itr_comp] * Js).sum()
                Qs[itr_comp] /= total_Js
                itr_comp += 1
        return Qs

    def comp_to_feat(self, phis_feat: np.ndarray, phis_comp: np.ndarray) -> None:
        itr_comp = 0
        for itr_feat in range(self._num_feat):
            phis_feat[itr_feat] = phis_comp[itr_comp]
            itr_comp += 1
            for _ in range(1, self._num_comp_per_feat[itr_feat]):
                phis_feat[itr_feat] += phis_comp[itr_comp]
                itr_comp += 1

    def volume_derivative(self, phis_comp: np.ndarray) -> np.ndarray:
        ans = np.zeros_like(phis_comp[0])
        for itr_comp in range(self.num_comp):
            ans -= phis_comp[itr_comp] / self._sizes[itr_comp]
        return ans


class IdealGasPolydispersedEntropy(IdealGasEntropyBase):
    r"""Class for entropic energy of mixture of polydispersed ideal gas.

    The particular form of dimensionless entropic energy reads

    .. math::
        f_\mathrm{entropy}(\{\phi_i\}) =
            \sum_{i=1}^{N_\mathrm{C}} \frac{\nu}{\nu_i}\phi_i \ln(\phi_i),

    where :math:`\phi_i` is the fraction of component :math:`i`. All components are
    assumed to have the same molecular volume :math:`\nu` by default. The relative
    molecular sizes :math:`l_i=\nu_i/\nu` can be changed by setting the optional parameter
    :paramref:`sizes`. Note that no implicit solvent is assumed.
    """

    def __init__(
        self,
        num_feat: int,
        sizes: np.ndarray | None = None,
        num_comp_per_feat: np.ndarray | int = 1,
    ):
        r"""
        Args:
            num_feat:
                Number of features :math:`N_\mathrm{S}`.
            sizes:
                The relative molecule volumes :math:`l_i = \nu_i/\nu` with respect to the
                volume of a reference molecule :math:`\nu`. It is treated as all-one
                vector by default.
            num_comp_per_feat:
                The number of components in each feature. An integer indicates that this
                value is the same for all features.
        """
        self.num_feat = num_feat

        num_comp_per_feat = convert_and_broadcast(num_comp_per_feat, (num_feat,))
        self._num_comp_per_feat = num_comp_per_feat

        super().__init__(
            num_comp=num_comp_per_feat.sum(),
            sizes=sizes,
        )
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def sizes(self) -> np.ndarray:
        r"""The relative molecule volumes :math:`l_i = \nu_i/\nu`."""
        return self._sizes

    @sizes.setter
    def sizes(self, sizes_new: np.ndarray):
        sizes_new = np.atleast_1d(sizes_new)
        shape = (self.num_comp,)
        self._sizes = np.array(np.broadcast_to(sizes_new, shape))

    def _compiled_impl(self) -> IdealGasPolydispersedEntropyCompiled:
        """Implementation of creating a compiled entropy instance.

        This method overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._compiled_impl` in
        :class:`~flory.entropy.base.EntropyBase`.

        Returns:
            : Instance of :class:`IdealGasPolydispersedEntropyCompiled`.
        """
        return IdealGasPolydispersedEntropyCompiled(
            self._sizes.astype(np.float64), self._num_comp_per_feat.astype(np.int32)
        )
