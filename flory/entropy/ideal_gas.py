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
    def __init__(self, sizes: np.ndarray):
        r"""The compiled ideal gas entropy.

        Args:
            sizes:
                The relative molecule volumes :math:`l_i = \nu_i/\nu`. The number of
                components is inferred from this matrix.
        """
        self._num_comp = sizes.shape[0]
        self._num_feat = sizes.shape[0]
        self._sizes = sizes  # do not affect chis

    @property
    def num_comp(self):
        return self._num_comp

    @property
    def num_feat(self):
        return self._num_feat

    @property
    def sizes(self):
        return self._sizes

    def partition(
        self, phis_comp: np.ndarray, omegas: np.ndarray, Js: np.ndarray
    ) -> np.ndarray:
        """Generate the partition function and fractions of components before normalization.
        Note that this function should modify :paramref:`phis_comp`

        Args:
            phis_comp:
                Output. The volume fractions of components before normalization
            omegas:
                Constant. The external field felt by the components
            Js:
                Constant. Volumes of compartments

        Returns:
            : The single molecule partition function :math:`Q_i`
        """
        Qs = np.zeros((self._num_comp,))
        total_Js = Js.sum()
        for itr_comp in range(self._num_comp):
            phis_comp[itr_comp] = np.exp(-omegas[itr_comp] * self._sizes[itr_comp])
            Qs[itr_comp] = (phis_comp[itr_comp] * Js).sum()
            Qs[itr_comp] /= total_Js
        return Qs

    def comp_to_feat(self, phis_feat: np.ndarray, phis_comp: np.ndarray):
        """Combine the fractions of species into fraction of features.
        Note that this function should modify :paramref:`phis_feat`

        Args:
            phis_feat:
                Output. the volume fractions of features
            phis_comp:
                Constant. the volume fractions of components
        """
        for itr_feat in range(self._num_feat):
            itr_comp = itr_feat
            phis_feat[itr_feat] = phis_comp[itr_comp]

    def volume_derivative(self, phis_comp: np.ndarray) -> np.ndarray:
        """Obtain the volume derivative of the partition function part of entropic energy

        Returns:
            : the volume derivative
        """
        ans = np.zeros_like(phis_comp[0])
        for itr_comp in range(self.num_comp):
            ans -= phis_comp[itr_comp] / self._sizes[itr_comp]
        return ans


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
            sizes = np.atleast_1d(sizes)
            shape = (num_comp,)
            self.sizes = np.array(np.broadcast_to(sizes, shape))

    def _compiled_impl(self) -> object:
        """make the entropy instance that can be used by the :mod:`mcmp` module."""

        return IdealGasEntropyCompiled(self.sizes)

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
        r"""returns Hessian :math:`\partial^2 f/\partial \phi^2` for the given composition

        Args:
            phis:
                The composition of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The full Hessian
        """
        return np.eye(self.num_comp) / (phis * self.sizes)[..., None]
