import logging
from typing import Optional, Union

import numpy as np

from ..commom import *

class FreeEnergyBase:
    def __init__(self, num_comp: int, sizes: Optional[np.ndarray] = None):
        """Base class for a general free energy of mixture.

        Args:
            num_comp:
                number of components in the mixture
            sizes:
                Relative molecule volumes of the components.
        """
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

    def _interaction_impl(self, **kwargs) -> object:
        """returns interaction instance containing necessary member functions for iteration.
        """
        raise NotImplementedError

    def _free_energy_density_impl(self, phis: np.ndarray) -> np.ndarray:
        """returns free energy for a given composition"""
        raise NotImplementedError

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""returns full Jacobian :math:`\partial f/\partial \phi` for the given composition"""
        raise NotImplementedError

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""returns full Hessian :math:`\partial^2 f/\partial \phi^2` for the given composition"""
        raise NotImplementedError

    def interaction(self, **kwargs_full) ->object:
        """Create the interaction instance containing necessary member functions for iteration
        This function requires the implementation of :meth:`_interaction_impl`. The
        interaction instance is a compiled class, which must implement following compiled
        functions:
        
        Detailed documentation required here.
        
        Args:
            kwargs_full:
                The keyword arguments which allow additional unused arguments.

        Returns:
            : The compiler interaction instance.
        """
        kwargs = filter_kwargs(kwargs_full, self._interaction_impl)
        return self._interaction_impl(**kwargs)

    def check_volume_fractions(self, phis: np.ndarray, axis: int = -1) -> np.ndarray:
        """Check whether volume fractions are valid.
        If the shape of :paramref:`phis` or it has non-positive values, an exception will be raised.
        Note that this function do not forbid volume fractions to be larger than 1.
        Args:
            phis: 
                Volume fractions of the components. Multiple compositions can be included.
            axis:
                The axis of the index of components. By the default the last dimension of
                :paramref:`phis` is considered as the index of components.

        Returns:
            : The volume fractions :paramref:`phis`
        """        
        phis = np.asanyarray(phis)

        if phis.shape[-1] != self.num_comp:
            self._logger.error(
                f"The shape of f{phis.shape} of volume fractions is incompatible with the number of components {self.num_comp}."
            )
            raise VolumeFractionError("Invalid size for volume fractions")

        if np.any(phis < 0):
            self._logger.error(f"Volume fractions {phis} contain negative values.")
            raise VolumeFractionError("Volume fractions must be all positive")
        return phis

    def free_energy_density(self, phis: np.ndarray) -> np.ndarray:
        """Calculate the free energy density for a given composition

        Args:
            phis:
                The composition of the phase(s). If multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : Free energy density of each phase.
        """
        phis = self.check_volume_fractions(phis)
        return self.free_energy_density(phis)

    def jacobian(self, phis: np.ndarray, index: Optional[int] = None) -> np.ndarray:
        """Calculate the Jacobian with/without volume conservation.
        If parameter :paramref:`index` is specified, the system will be considered as
        conserved and the volume fraction of component :paramref:`index` is treated to be
        not independent. Note that different from :meth:`exchange_chemical_potentials`,
        :meth:`jacobian` removed the dependent variable completely, while
        :meth:`exchange_chemical_potentials` keeps the exchange chemical potential of the
        component :paramref:`index`. Pass `None` to :paramref:`index` indicates the system
        is not conserved.

        Args:
            phis:
                The composition of the phase(s). If multiple phases are included, the
                index of the components must be the last dimension.
            index:
                Index of the dependent component. `None` indicates the system is not conserved.

        Returns:
            : Jacobian of each phase with/without volume conservation.
        """
        phis = self.check_volume_fractions(phis)
        j_full = self._jacobian_impl(phis)
        if index is None:
            return j_full
        else:
            return (
                np.delete(j_full, index, axis=-1) - j_full[..., index, None]
            )  # chain rule

    def hessian(self, phis: np.ndarray, index: Optional[int] = None) -> np.ndarray:
        """Calculate the Hessian with/without volume conservation.
        If parameter :paramref:`index` is specified, the system will be considered as
        conserved and the volume fraction of component :paramref:`index` is treated to be
        not independent. Note that different from :meth:`exchange_chemical_potentials`,
        :meth:`hessian` removed the dependent variable completely, while
        :meth:`exchange_chemical_potentials` keeps the exchange chemical potential of the
        component :paramref:`index`. Pass `None` to :paramref:`index` indicates the system
        is not conserved.

        Args:
            phis:
                The composition of the phase(s). If multiple phases are included, the
                index of the components must be the last dimension.
            index:
                Index of the dependent component. `None` indicates the system is not conserved.

        Returns:
            : The Hessian with/without volume conservation.
        """
        phis = self.check_volume_fractions(phis)
        h_full = self._hessian_impl(phis)
        if index is None:
            return h_full
        else:
            h_reduced_full = (
                h_full
                - h_full[..., index, None, :]
                - h_full[..., :, index, None]
                + h_full[..., index, None, index, None]
            )  # chain rule

            return np.delete(np.delete(h_reduced_full, index, axis=-1), index, axis=-2)

    def chemical_potentials(self, phis: np.ndarray) -> np.ndarray:
        """Calculate original chemical potentials

        Args:
            phis:
                The composition of the phase(s). If multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The original chemical potentials.
        """
        f = self.free_energy_density(phis)
        j = self.jacobian(phis)
        return np.atleast_1d(f)[..., None] - np.einsum("...i,...i->...", phis, j) + j

    def exchange_chemical_potentials(self, phis: np.ndarray, index: int) -> np.ndarray:
        """Calculate exchange chemical potentials, treating component `index` as the solvent.
        The exchange chemical potentials is obtained by removing chemical potential of the
        solvent. The exchange chemical potential of the solvent is always zero and kept in
        the result. The nonzero values are identical to the conserved Jacobian, see
        :meth:`jacobian` for more information.

        Args:
            phis:
                The composition of the phase(s). If multiple phases are included, the
                index of the components must be the last dimension.
            index:
                Index of the solvent component

        Returns:
            : The exchange chemical potentials of component with respect to the solvent component.
        """
        mus = self.chemical_potentials(phis)
        return mus - mus[..., index, None]

    def pressure(self, phis: np.ndarray, index: int) -> np.ndarray:
        """Calculate osmotic pressure of the solvent by treating component `index` as the solvent.
        The osmotic pressure of the solvent is proportional to the original chemical
        potential of the solvent.

        Args:
            phis:
                The composition of the phase(s). If multiple phases are included, the
                index of the components must be the last dimension.
            index:
                Index of the solvent component

        Returns:
            : The osmotic pressure of the solvent component `index`.
        """
        mus = self.chemical_potentials(phis)
        return -mus[..., index] / self.sizes[index]

    def num_unstable_modes(
        self, phis: np.ndarray, conserved: bool = True
    ) -> Union[int, np.ndarray]:
        """Count the number of unstable modes with/without volume conservation.

        Args:
            phis:
                The composition of the phase(s). If multiple phases are included, the
                index of the components must be the last dimension.
            conserved:
                Whether the system conserves volume. If `True`, the first component is
                considered as the dependent on when calculating the Hessian. See
                :meth:`hessian` for more information.

        Returns:
            : The number of negative eigenvalues of the Hessian.
        """
        eigenvalues = np.linalg.eigvalsh(self.hessian(phis, 0 if conserved else None))
        return np.sum(eigenvalues < 0, axis=-1).astype(int)

    def is_stable(
        self, phis: np.ndarray, conserved: bool = True
    ) -> Union[int, np.ndarray]:
        """Determine whether the mixture is locally stable.

        Args:
            phis:
                The composition of the phase(s). If multiple phases are included, the
                index of the components must be the last dimension.
            conserved:
                Whether the system conserves volume. If `True`, the first component is
                considered as the dependent on when calculating the Hessian. See
                :meth:`hessian` for more information.

        Returns:
            : The number of negative eigenvalues of the Hessian.
        """
        return self.num_unstable_modes(phis, conserved) == 0

