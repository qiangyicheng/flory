"""Module for ideal gas entropic energy of mixture.

"""

from typing import Optional, Tuple
import logging

from numba.experimental import jitclass
from numba import float64, int32
import numpy as np
from .base import EntropyBase


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_num_feat", int32),  # a scalar
        ("_sizes", float64[::1]),  # a C-continuous array
    ]
)
class IdealGasEntropyCompiled(object):
    r"""Compiled class for the entropic energy of mixture of ideal gas.
    For ideal gas, the Boltzmann factor :math:`p_i^{(m)}` is determined by the
    relative volumes of the molecules :math:`l_i = \nu_i/\nu` and the external fields it
    feel, :math:`w_r^{(m)}`,

        .. math::
            p_i^{(m)} = \exp(-l_i w_r^{(m)}).

    Note that this class assumes that there's no degeneracy of the components' features,
    namely all components have their own external fields. Therefore the number of features
    :math:`N_\mathrm{s}` are the same as the number of components :math:`N_\mathrm{c}`.
    """

    def __init__(self, sizes: np.ndarray):
        r"""
        Args:
            sizes:
                The relative molecule volumes :math:`l_i = \nu_i/\nu`. The number of
                components :math:`N_\mathrm{c}` is inferred from this matrix. The number
                of features :math:`N_\mathrm{s}` is set to be same as
                :math:`N_\mathrm{c}`.
        """
        self._num_comp = sizes.shape[0]
        self._num_feat = sizes.shape[0]
        self._sizes = sizes

    @property
    def num_comp(self):
        r"""Number of components :math:`N_\mathrm{c}`."""
        return self._num_comp

    @property
    def num_feat(self):
        r"""Number of features :math:`N_\mathrm{s}`."""
        return self._num_feat

    @property
    def sizes(self):
        r"""The relative molecule volumes :math:`l_i = \nu_i/\nu`."""

        return self._sizes

    def partition(
        self, phis_comp: np.ndarray, omegas: np.ndarray, Js: np.ndarray
    ) -> np.ndarray:
        """Calculate the partition function and Boltzmann factors.
        The Boltzmann factors are equivalent to the volume fractions of components before
        normalization. Note that this function should modify :paramref:`phis_comp` directly.

        Args:
            phis_comp:
                Output. The Boltzmann factors, namely the volume fractions of components
                before normalization.
            omegas:
                Constant. The external field felt by the components.
            Js:
                Constant. Volumes of compartments.

        Returns:
            : The single molecule partition function of components :math:`Q_i`.
        """
        Qs = np.zeros((self._num_comp,))
        total_Js = Js.sum()
        for itr_comp in range(self._num_comp):
            phis_comp[itr_comp] = np.exp(-omegas[itr_comp] * self._sizes[itr_comp])
            Qs[itr_comp] = (phis_comp[itr_comp] * Js).sum()
            Qs[itr_comp] /= total_Js
        return Qs

    def comp_to_feat(self, phis_feat: np.ndarray, phis_comp: np.ndarray):
        """Combine the fractions of components into fractions of features.
        Note that this function should modify :paramref:`phis_feat` directly.

        Args:
            phis_feat:
                Output. The volume fractions of features.
            phis_comp:
                Constant. The volume fractions of components.
        """
        for itr_feat in range(self._num_feat):
            itr_comp = itr_feat
            phis_feat[itr_feat] = phis_comp[itr_comp]

    def volume_derivative(self, phis_comp: np.ndarray) -> np.ndarray:
        """Obtain the volume derivatives of the partition function part of entropic energy.

        Returns:
            : The volume derivatives.
        """
        ans = np.zeros_like(phis_comp[0])
        for itr_comp in range(self.num_comp):
            ans -= phis_comp[itr_comp] / self._sizes[itr_comp]
        return ans


class IdealGasEntropy(EntropyBase):
    r"""Class for entropic energy of mixture of ideal gas.
    The particular form of entropic energy reads
    
    .. math::
        f(\{\phi_i\}) = \frac{k_\mathrm{B}T}{\nu}
            \sum_{i=1}^{N_\mathrm{c}} \frac{\nu}{\nu_i}\phi_i \ln(\phi_i),

    where :math:`\phi_i` is the fraction of component :math:`i`. All components are
    assumed to have the same molecular volume :math:`\nu` by default. The relative
    molecular sizes :math:`l_i=\nu_i/\nu` can be changed by setting the optional parameter
    :paramref:`sizes`. Note that no implicit solvent is assumed.
    """

    def __init__(
        self,
        num_comp: int,
        sizes: Optional[np.ndarray] = None,
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`.
            sizes:
                The relative molecule volumes :math:`l_i = \nu_i/\nu` with respect to the
                volume of a reference molecule :math:`\nu`. It is treated as all-one
                vector by default.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)
        if sizes is None:
            self.sizes = np.ones(num_comp)
        else:
            sizes = np.atleast_1d(sizes)
            shape = (num_comp,)
            self.sizes = np.array(np.broadcast_to(sizes, shape))

    def _compiled_impl(self) -> object:
        """Implementation of creating a compiled entropy instance.
        This function overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._compiled_impl` in
        :class:`~flory.entropy.base.EntropyBase`. 
        
        Returns:
            : Instance of :class:`IdealGasEntropyCompiled`.
        """

        return IdealGasEntropyCompiled(self.sizes)

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        """Implementation of calculating entropic energy.
        This function overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._energy_impl` in
        :class:`~flory.entropy.base.EntropyBase`.

        Args:
            phis:
                The volume fractions of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The entropic energy density.
        """
        return np.einsum("...i,...i->...", phis / self.sizes, np.log(phis))

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Jacobian :math:`\partial f/\partial \phi`.
        This function overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._jacobian_impl` in
        :class:`~flory.entropy.base.EntropyBase`.

        Args:
            phis:
                The volume fractions of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The full Jacobian.
        """
        return np.log(phis) / self.sizes + 1.0 / self.sizes

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Hessian :math:`\partial^2 f/\partial \phi^2`.
        This function overwrites the interface
        :meth:`~flory.entropy.base.EntropyBase._hessian_impl` in
        :class:`~flory.entropy.base.EntropyBase`.

        Args:
            phis:
                The volume fractions of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The full Hessian.
        """
        return np.eye(self.num_comp) / (phis * self.sizes)[..., None]
