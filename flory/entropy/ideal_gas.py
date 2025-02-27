"""Module for ideal gas entropic energy of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np
from numba import float64, int32
from numba.experimental import jitclass

from ..common.math import xlogx
from .base import EntropyBase, EntropyBaseCompiled


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_num_feat", int32),  # a scalar
        ("_sizes", float64[::1]),  # a C-continuous array
    ]
)
class IdealGasEntropyCompiled(EntropyBaseCompiled):
    r"""Compiled class for the entropic energy of mixture of ideal gas.

    For ideal gas, the Boltzmann factor :math:`p_i^{(m)}` is determined by the
    relative volumes of the molecules :math:`l_i = \nu_i/\nu` and the mean fields it
    feel, :math:`w_r^{(m)}`,

        .. math::
            p_i^{(m)} = \exp(-l_i w_r^{(m)}).

    Note that this class assumes that there's no degeneracy of the components' features,
    in other words all components have their own mean fields, such that the component
    index is the same as the feature id, :math:`i=r`. Therefore the number of features
    :math:`N_\mathrm{S}` are the same as the number of components :math:`N_\mathrm{C}`.
    """

    def __init__(self, sizes: np.ndarray):
        r"""
        Args:
            sizes:
                1D array with the size of :math:`N_\mathrm{C}`, containing the relative
                molecule volumes :math:`l_i = \nu_i/\nu`. The number of components
                :math:`N_\mathrm{C}` is inferred from this array. The number of features
                :math:`N_\mathrm{S}` is set to be same as :math:`N_\mathrm{C}`.
        """
        self._num_comp = sizes.shape[0]
        self._num_feat = sizes.shape[0]
        self._sizes = sizes

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
        for itr_comp in range(self._num_comp):
            phis_comp[itr_comp] = np.exp(-omegas[itr_comp] * self._sizes[itr_comp])
            Qs[itr_comp] = (phis_comp[itr_comp] * Js).sum()
            Qs[itr_comp] /= total_Js
        return Qs

    def comp_to_feat(self, phis_feat: np.ndarray, phis_comp: np.ndarray) -> None:
        for itr_feat in range(self._num_feat):
            itr_comp = itr_feat
            phis_feat[itr_feat] = phis_comp[itr_comp]

    def volume_derivative(self, phis_comp: np.ndarray) -> np.ndarray:
        ans = np.zeros_like(phis_comp[0])
        for itr_comp in range(self.num_comp):
            ans -= phis_comp[itr_comp] / self._sizes[itr_comp]
        return ans


class IdealGasEntropyBase(EntropyBase):
    r"""Class for entropic energy of mixture of ideal gas.

    The particular form of dimensionless entropic energy reads

    .. math::
        f_\mathrm{entropy}(\{\phi_i\}) =
            \sum_{i=1}^{N_\mathrm{C}} \frac{\nu}{\nu_i}\phi_i \ln(\phi_i),

    where :math:`\phi_i` is the fraction of component :math:`i`. All components are
    assumed to have the same molecular volume :math:`\nu` by default. The relative
    molecular sizes :math:`l_i=\nu_i/\nu` can be changed by setting the optional parameter
    :paramref:`sizes`. Note that no implicit solvent is assumed. This class only
    implements the common methods of all ideal gas entropy. Note that
    :meth:`~flory.entropy.base.EntropyBase._compiled_impl` is not implemented.
    """

    def __init__(
        self,
        num_comp: int,
        sizes: np.ndarray | None = None,
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            sizes:
                The relative molecule volumes :math:`l_i = \nu_i/\nu` with respect to the
                volume of a reference molecule :math:`\nu`. It is treated as all-one
                vector by default.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)
        if sizes is None:
            self._sizes = np.ones(num_comp)
        else:
            sizes = np.atleast_1d(sizes)
            shape = (num_comp,)
            self._sizes = np.array(np.broadcast_to(sizes, shape))

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating entropic energy :math:`f_\mathrm{entropy}`.

        This method overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._energy_impl` in
        :class:`~flory.entropy.base.EntropyBase`.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The entropic energy density.
        """
        return np.sum(xlogx(phis) / self._sizes, axis=-1)

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Jacobian :math:`\partial f_\mathrm{entropy}/\partial \phi_i`.

        This method overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._jacobian_impl` in
        :class:`~flory.entropy.base.EntropyBase`.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The full Jacobian.
        """
        return np.log(phis) / self._sizes + 1.0 / self._sizes

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Hessian :math:`\partial^2 f_\mathrm{entropy}/\partial \phi_i^2`.

        This method overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._hessian_impl` in
        :class:`~flory.entropy.base.EntropyBase`.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The full Hessian.
        """
        # manually generate the matrix sine we assume no prior knowledge for the dimension of `phis``
        ans_diag = 1.0 / phis * self._sizes
        shape = list(ans_diag.shape)
        shape.append(self.num_comp)
        ans = np.zeros(shape)
        for itr in range(self.num_comp):
            ans[..., itr, itr] = ans_diag[..., itr]
        return ans

    def _jacobian_fractions_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating :math:`\phi_i \partial f_\mathrm{entropy}/\partial \phi_i`.

        This method is not included in the interface, since it is only needed when
        :math:`\phi_i \partial f_\mathrm{entropy}/\partial \phi_i` exist but
        :math:`\partial f_\mathrm{entropy}/\partial \phi_i` diverges.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The full Jacobian.
        """
        return xlogx(phis) / self._sizes + phis / self._sizes


class IdealGasEntropy(IdealGasEntropyBase):
    r"""Class for entropic energy of mixture of ideal gas.

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
        num_comp: int,
        sizes: np.ndarray | None = None,
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            sizes:
                The relative molecule volumes :math:`l_i = \nu_i/\nu` with respect to the
                volume of a reference molecule :math:`\nu`. It is treated as all-one
                vector by default.
        """
        super().__init__(num_comp, sizes)
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

    def _compiled_impl(self) -> IdealGasEntropyCompiled:
        """Implementation of creating a compiled entropy instance.

        This method overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._compiled_impl` in
        :class:`~flory.entropy.base.EntropyBase`.

        Returns:
            : Instance of :class:`IdealGasEntropyCompiled`.
        """

        return IdealGasEntropyCompiled(self._sizes.astype(np.float64))
