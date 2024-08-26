"""Module for Flory-Huggins interaction energy of mixture.
"""

import logging
from typing import Any, Union, Optional

from numba.experimental import jitclass
from numba import float64, int32
import numpy as np
from .base import InteractionBase


@jitclass(
    [
        ("_num_feat", int32),  # a scalar
        ("_chis", float64[:, ::1]),  # a C-continuous array
        ("_incomp_coef", float64),  # a scalar
    ]
)
class FloryHugginsInteractionCompiled(object):
    r"""Compiled class for Flory-Huggins interaction energy.
    Flory-Huggins interaction is the second-ordered interaction, whose energy reads,

    .. math::
        f(\{\phi_i\}) = \frac{k_\mathrm{B}T}{\nu}
                \!\sum_{i,j=1}^{N_\mathrm{s}} \frac{\chi_{ij}}{2} \phi_i\phi_j

    Note that here we use describe the system by features.

    """

    def __init__(self, chis: np.ndarray, chis_shift: float):
        r"""
        Args:
            chis:
                The Flory-Huggins matrix :math:`\chi_{ij}`. The number of features is
                inferred from this matrix.
            chis_shift:
                The shift of entire Flory-Huggins matrix for the calculation.
        """
        self._num_feat = chis.shape[0]
        self._chis = chis  # do not affect chis
        self._incomp_coef = chis.sum() + chis_shift * self._num_feat * self._num_feat

    @property
    def num_feat(self):
        r"""Number of features :math:`N_\mathrm{s}`."""
        return self._num_feat

    @property
    def chis(self):
        r"""Flory-Huggins parameters for features :math:`\chi_{ij}`."""
        return self._chis

    def energy(self, potential: np.ndarray, phis_feat: np.ndarray) -> np.ndarray:
        r"""Calculate the Flory-Huggins interaction energy.

        Args:
            potential:
                Constant. The fields :math:`w_r^{(m)}` that the features felt.
                Usually this is the returned value of :meth:`potential`. This parameter is
                passed in since usually the calculation of energy can be accelerated by
                directly using the potential.
            phis_feat:
                Constant. Volume fractions of features :math:`\phi_i^{(m)}`.

        Returns:
            : The interaction energy.
        """
        # since Flory-Huggins free energy contains only 2nd-ordered interactions,
        # the interaction energy is directly calculated from potential and phis
        ans = np.zeros_like(potential[0])
        for itr in range(self._num_feat):
            ans += potential[itr] * phis_feat[itr]
        ans *= 0.5
        return ans

    def potential(self, phis_feat: np.ndarray) -> np.ndarray:
        r"""Calculate the potential :math:`w_r^{(m)}` that the features feel.

        Args:
            phis_feat:
                Constant. Volume fractions of features :math:`\phi_i^{(m)}`.

        Returns:
            : The potential :math:`w_r^{(m)}`.
        """
        return self._chis @ phis_feat

    def incomp_coef(self, phis_feat: np.ndarray) -> float:
        r"""Calculate the coefficient for incompressibility.

        Args:
            phis_feat:
                Constant. Volume fractions of features :math:`\phi_i^{(m)}`.

        Returns:
            float: The coefficient for incompressibility.
        """
        return self._incomp_coef


class FloryHugginsInteraction(InteractionBase):
    r"""Class for Flory-Huggins interaction energy of mixture.
    The particular form of interaction energy reads

        .. math::
            f(\{\phi_i\}) = \frac{k_\mathrm{B}T}{\nu}
                \!\sum_{i,j=1}^{N_\mathrm{c}} \frac{\chi_{ij}}{2} \phi_i\phi_j

        where :math:`\phi_i` is the fraction of component :math:`i`.
    """

    def __init__(
        self,
        num_comp: int,
        chis: Union[np.ndarray, float],
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`.
            chis:
                The Flory-Huggins interaction matrix of components :math:`\chi_{ij}`.
        """
        super().__init__(num_comp=num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)

        chis = np.atleast_1d(chis)

        shape = (num_comp, num_comp)
        chis = np.array(np.broadcast_to(chis, shape))

        # ensure that the chi matrix is symmetric
        if not np.allclose(chis, chis.T):
            self._logger.warning("Using symmetrized χ interaction-matrix")
        self.chis = 0.5 * (chis + chis.T)

    @property
    def independent_entries(self) -> np.ndarray:
        r"""Entries of the upper triangle of the :math:`\chi_{ij}` matrix"""
        return self.chis[np.triu_indices_from(self.chis, k=0)]

    @classmethod
    def from_uniform(
        cls,
        num_comp: int,
        chi: float,
        *,
        vanishing_diagonal: bool = True,
    ):
        r"""Create Flory-Huggins interaction with uniform :math:`\chi_{ij}` matrix.

        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`.
            chi:
                The value of all non-zero values in the interaction matrix :math:`\chi{i
                \ne j}`
            vanishing_diagonal:
                Whether the diagonal elements of the :math:`\chi_{ij}` matrix are set to
                be zero.
        """
        obj = cls(num_comp, 0)
        obj.set_uniform_chis(chi, vanishing_diagonal=vanishing_diagonal)
        return obj

    @classmethod
    def from_random_normal(
        cls,
        num_comp: int,
        chi_mean: float = 0,
        chi_std: float = 1,
        *,
        vanishing_diagonal: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        r"""Create Flory-Huggins interaction with random :math:`\chi_{ij}` matrix.

        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`.
            chi_mean:
                Mean interaction :math:`\bar{\chi}`.
            chi_std:
                Standard deviation of the interactions :math:`\sigma_{\chi}`.
            vanishing_diagonal:
                Whether the diagonal elements of the :math:`\chi_{ij}` matrix are set to
                be zero.
            rng:
                The random number generator.

        """
        obj = cls(num_comp, 0)
        obj.set_random_chis(
            chi_mean, chi_std, vanishing_diagonal=vanishing_diagonal, rng=rng
        )
        return obj

    def set_uniform_chis(
        self,
        chi: float,
        *,
        vanishing_diagonal: bool = True,
    ):
        r"""Set Flory-Huggins interaction with uniform :math:`\chi_{ij}` matrix.

        Args:
            chi:
                The value of all non-zero values in the interaction matrix :math:`\chi{i
                \ne j}`
            vanishing_diagonal:
                Whether the diagonal elements of the :math:`\chi_{ij}` matrix are set to
                be zero.
        """
        self.chis = np.full((self.num_comp, self.num_comp), chi)
        if vanishing_diagonal:
            self.chis[np.diag_indices_from(self.chis)] = 0

    def set_random_chis(
        self,
        chi_mean: float = 0,
        chi_std: float = 1,
        *,
        vanishing_diagonal: bool = True,
        rng=None,
    ):
        r"""Set Flory-Huggins interaction with random :math:`\chi_{ij}` matrix.

        Args:
            chi_mean:
                Mean interaction :math:`\bar{\chi}`.
            chi_std:
                Standard deviation of the interactions :math:`\sigma_{\chi}`.
            vanishing_diagonal:
                Whether the diagonal elements of the :math:`\chi_{ij}` matrix are set to
                be zero.
            rng:
                The random number generator.
        """
        if rng is None:
            rng = np.random.default_rng()

        self.chis[:] = 0  # reset old values

        # determine random entries
        if vanishing_diagonal:
            num_entries = self.num_comp * (self.num_comp - 1) // 2
        else:
            num_entries = self.num_comp * (self.num_comp + 1) // 2
        chi_vals = rng.normal(chi_mean, chi_std, num_entries)

        # build symmetric  matrix from this
        i, j = np.triu_indices(self.num_comp, 1 if vanishing_diagonal else 0)
        self.chis[i, j] = chi_vals
        self.chis[j, i] = chi_vals

    def _compiled_impl(self, *, additional_chis_shift: float = 1.0) -> object:
        """Implementation of creating a compiled interaction instance.
        This function overwrites the interface
        :meth:`~flory.entropy.base.InteractionBase._compiled_impl` in
        :class:`~flory.entropy.base.InteractionBase`. 

        Args:
            additional_chis_shift:
                Shift of the entire chis matrix to improve the convergence by evolving
                towards incompressible system faster. This value should be larger than 0.
                This value only affects the numerics, not the actual physical system. Note
                that with very large value, the convergence will be slowed down, since the
                algorithm no longer have enough ability to temporarily relax the
                incompressibility.
                
        Returns:
            : Instance of :class:`FloryHugginsInteractionCompiled`.
        """

        return FloryHugginsInteractionCompiled(
            self.chis, -self.chis.min() + additional_chis_shift
        )

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        """Implementation of calculating interaction energy.
        This function overwrites the interface
        :meth:`~flory.entropy.base.InteractionBase._energy_impl` in
        :class:`~flory.entropy.base.InteractionBase`.

        Args:
            phis:
                The composition of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.

        Returns:
            : The interaction energy density
        """
        ans = 0.5 * np.einsum("...i,...j,ij->...", phis, phis, self.chis)
        return ans

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Jacobian :math:`\partial f/\partial \phi`.
        This function overwrites the interface
        :meth:`~flory.entropy.base.InteractionBase._jacobian_impl` in
        :class:`~flory.entropy.base.InteractionBase`.

        Args:
            phis:
                The composition of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.
        Returns:
            : The full Jacobian
        """
        ans = phis @ self.chis
        return ans

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Hessian :math:`\partial^2 f/\partial \phi^2`.
        This function overwrites the interface
        :meth:`~flory.entropy.base.InteractionBase._hessian_impl` in
        :class:`~flory.entropy.base.InteractionBase`.

        Args:
            phis:
                The composition of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.
        Returns:
            : The full Hessian
        """
        return np.zeros_like(phis)[..., None] + self.chis  # type: ignore
