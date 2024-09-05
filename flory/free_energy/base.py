"""Module for a general free energy of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np

from ..common import *
from ..entropy import EntropyBase
from ..interaction import InteractionBase


class FreeEnergyBase:
    """Base class for a general free energy of mixture.

    A free energy is constructed by an interactions energy and a entropic energy. Once the
    energy, Jacobian and Hessian of both interactions energy and entropic energy are
    implemented, class :class:`FreeEnergyBase` provides methods such as the chemical
    potential of the components.
    """

    def __init__(self, interaction: InteractionBase, entropy: EntropyBase):
        """
        Args:
            interaction:
                The interaction energy instance.
            entropy:
                The entropic energy instance.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        if interaction.num_comp != entropy.num_comp:
            self._logger.error(
                "Interactions requires %d components while entropy requires %d.",
                interaction.num_comp,
                entropy.num_comp,
            )
            raise ComponentNumberError(
                "Number of component mismatch between interaction and entropy."
            )
        self.interaction = interaction
        self.entropy = entropy
        self.num_comp = interaction.num_comp

    def interaction_compiled(self, **kwargs_full) -> InteractionBase:
        """Get the compiled instance of the interaction.

        Args:
            kwargs_full:
                The keyword arguments for method
                :meth:`~flory.interaction.base.InteractionBase.compiled` of the
                interaction instance but allowing redundant arguments.

        """
        return self.interaction.compiled(**kwargs_full)

    def entropy_compiled(self, **kwargs_full) -> EntropyBase:
        """Get the compiled instance of the entropy.

        Args:
            kwargs_full:
                The keyword arguments for method
                :meth:`~flory.entropy.base.EntropyBase.compiled` of the
                entropy instance but allowing redundant arguments.

        """

        return self.entropy.compiled(**kwargs_full)

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating free energy :math:`f`.

        This method is general, thus does not need to be overwritten. The method makes use
        of :meth:`~flory.interaction.base.InteractionBase._energy_impl` in
        :class:`~flory.interaction.base.InteractionBase` and
        :meth:`~flory.entropy.base.EntropyBase._energy_impl` in
        :class:`~flory.entropy.base.EntropyBase`. Consider define custom interaction or
        entropy if a custom free energy is needed.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The free energy density.
        """
        return self.interaction._energy_impl(phis) + self.entropy._energy_impl(phis)

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Jacobian :math:`\partial f/\partial \phi_i`.

        This method is general, thus does not need to be overwritten. The method makes use
        of :meth:`~flory.interaction.base.InteractionBase._jacobian_impl` in
        :class:`~flory.interaction.base.InteractionBase` and
        :meth:`~flory.entropy.base.EntropyBase._jacobian_impl` in
        :class:`~flory.entropy.base.EntropyBase`. Consider define custom interaction or
        entropy if a custom free energy is needed.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The The full Jacobian.
        """
        return self.interaction._jacobian_impl(phis) + self.entropy._jacobian_impl(phis)

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Hessian :math:`\partial^2 f/\partial \phi_i^2`.

        This method is general, thus does not need to be overwritten. The method makes use
        of :meth:`~flory.interaction.base.InteractionBase._hessian_impl` in
        :class:`~flory.interaction.base.InteractionBase` and
        :meth:`~flory.entropy.base.EntropyBase._hessian_impl` in
        :class:`~flory.entropy.base.EntropyBase`. Consider define custom interaction or
        entropy if a custom free energy is needed.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The full Hessian.
        """
        return self.interaction._hessian_impl(phis) + self.entropy._hessian_impl(phis)

    def check_volume_fractions(self, phis: np.ndarray, axis: int = -1) -> np.ndarray:
        r"""Check whether volume fractions are valid.

        If the shape of :paramref:`phis` or it has non-positive values, an exception will be raised.
        Note that this method does not forbid volume fractions to be larger than 1.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
            axis:
                The axis of the index of components. By the default the last dimension of
                :paramref:`phis` is considered as the index of components.

        Returns:
            : The volume fractions :paramref:`phis`
        """
        phis = np.asanyarray(phis)

        if phis.shape[axis] != self.num_comp:
            self._logger.error(
                "The shape %s of volume fractions is incompatible with the number of components %d.",
                phis.shape,
                self.num_comp,
            )
            raise VolumeFractionError("Invalid size for volume fractions")

        if np.any(phis < 0):
            self._logger.error("Volume fractions %s contain negative values.", phis)
            raise VolumeFractionError("Volume fractions must be all positive")
        return phis

    def free_energy_density(self, phis: np.ndarray) -> np.ndarray:
        r"""Calculate the free energy density.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : Free energy density of each phase.
        """
        phis = self.check_volume_fractions(phis)
        return self._energy_impl(phis)

    def jacobian(self, phis: np.ndarray, index: int | None = None) -> np.ndarray:
        r"""Calculate the Jacobian with/without volume conservation.

        If parameter :paramref:`index` is specified, the system will be considered as
        conserved and the volume fraction of component :paramref:`index` is treated to be
        not independent. Note that different from :meth:`exchange_chemical_potentials`,
        :meth:`jacobian` removed the dependent variable completely, while
        :meth:`exchange_chemical_potentials` keeps the exchange chemical potential of the
        component :paramref:`index`. Pass `None` to :paramref:`index` indicates the system
        is not conserved.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
            index:
                Index of the dependent component. By default the system is not conserved.

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

    def hessian(self, phis: np.ndarray, index: int | None = None) -> np.ndarray:
        r"""Calculate the Hessian with/without volume conservation.

        If parameter :paramref:`index` is specified, the system will be considered as
        conserved and the volume fraction of component :paramref:`index` is treated to be
        not independent. Note that different from :meth:`exchange_chemical_potentials`,
        :meth:`hessian` removed the dependent variable completely, while
        :meth:`exchange_chemical_potentials` keeps the exchange chemical potential of the
        component :paramref:`index`. Pass `None` to :paramref:`index` indicates the system
        is not conserved.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
            index:
                Index of the dependent component. By default the system is not conserved.

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
        r"""Calculate original chemical potentials by unit volume.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The original chemical potentials.
        """
        f = self.free_energy_density(phis)
        j = self.jacobian(phis)
        ans = (
            np.atleast_1d(f)[..., None]
            - np.einsum("...i,...i->...", phis, j)[..., None]
            + j
        )
        return ans

    def exchange_chemical_potentials(self, phis: np.ndarray, index: int) -> np.ndarray:
        r"""Calculate exchange chemical potentials.

        Component :paramref:`index` is treated as the solvent. The exchange chemical
        potentials is obtained by removing chemical potential of the solvent. The exchange
        chemical potential of the solvent is always zero and kept in the result. The
        nonzero values are identical to the conserved Jacobian, see :meth:`jacobian` for
        more information.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
            index:
                Index of the solvent component

        Returns:
            : The exchange chemical potentials of component with respect to the solvent component.
        """
        mus = self.chemical_potentials(phis)
        return mus - mus[..., index, None]

    def pressure(self, phis: np.ndarray, index: int) -> np.ndarray:
        r"""Calculate osmotic pressure of the solvent.

        Component :paramref:`index` is treated as the solvent. The osmotic pressure of the
        solvent is proportional to the original chemical potential of the solvent.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
            index:
                Index of the solvent component

        Returns:
            : The osmotic pressure of the solvent component `index`.
        """
        mus = self.chemical_potentials(phis)
        return -mus[..., index]

    def num_unstable_modes(
        self, phis: np.ndarray, conserved: bool = True
    ) -> int | np.ndarray:
        r"""Count the number of unstable modes with/without volume conservation.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
            conserved:
                Whether the system conserves volume. If `True`, the first component is
                considered as the dependent on when calculating the Hessian. See
                :meth:`hessian` for more information.

        Returns:
            : The number of negative eigenvalues of the Hessian.
        """
        eigenvalues = np.linalg.eigvalsh(self.hessian(phis, 0 if conserved else None))
        return np.sum(eigenvalues < 0, axis=-1).astype(int)

    def is_stable(self, phis: np.ndarray, conserved: bool = True) -> int | np.ndarray:
        r"""Determine whether the mixture is locally stable.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
            conserved:
                Whether the system conserves volume. If `True`, the first component is
                considered as the dependent on when calculating the Hessian. See
                :meth:`hessian` for more information.

        Returns:
            : The number of negative eigenvalues of the Hessian.
        """
        return self.num_unstable_modes(phis, conserved) == 0
