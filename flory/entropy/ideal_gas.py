"""
Module defining thermodynamic quantities of multicomponent phase separation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Union, Optional
import logging

from numba.experimental import jitclass
from numba import float64, int32
import numpy as np
from .base import EntropyBase


class IdealGasEntropy(EntropyBase):
    r"""represents the entropy of a multicomponent mixture of ideal gas

    The particular implementation of the entropic energy density reads

    .. math::
        f(\{\phi_i\}) = \frac{k_\mathrm{B}T}{\nu}
            \sum_{i=1}^N \frac{\nu}{\nu_i}\phi_i \ln(\phi_i)

    where :math:`\phi_i` is the fraction of component :math:`i`. All components are
    assumed to have the same molecular volume :math:`\nu` by default. The relative
    molecular sizes :math:`l_i=\nu_i/\nu` can be changed by setting the optional parameter
    `sizes`. Note that no implicit solvent is assumed.
    """

    def __init__(
        self,
        num_comp: int,
        sizes: Optional[np.ndarray] = None,
    ):
        """
        Args:
            num_comp:
                Number of components in the system
            sizes:
                The relative volumes with respect to the volume of an imaginary reference
                molecular. It is treated as all-one vector by default or passing `None`.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)
        self.num_comp = num_comp
        if sizes is None:
            self.sizes = np.ones(num_comp)
        else:
            self.sizes = np.array(sizes)
            if self.sizes.shape != (self.num_comp,):
                self._logger.error(
                    f"sizes vector with size of {self.sizes.shape} is invalid for {self.num_comp}-component system."
                )
                raise ValueError("length of explicit sizes vector must be the num_comp.")

    def _compiled_impl(self) -> object:
        """make the entropy instance that can be used by the :mod:`mcmp` module."""

        raise NotImplementedError

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        """returns entropic energy for a given composition

        Args:
            phis:
                The composition of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The entropic energy density
        """
        return np.einsum("...i,...i->...", phis / self.sizes, np.log(phis))

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""returns full Jacobian :math:`\partial f/\partial \phi` for the given composition

        Args:
            phis:
                The composition of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The full Hessian
        """
        return np.log(phis) / self.sizes + 1.0 / self.sizes

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        """returns Hessian :math:`\partial^2 f/\partial \phi^2` for the given composition

        Args:
            phis:
                The composition of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The full Hessian
        """
        return np.eye(self.num_comp) / (phis * self.sizes)[..., None]
