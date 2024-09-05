"""Module for Flory-Huggins interaction energy of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np
from numba import float64, int32
from numba.experimental import jitclass

from .base import InteractionBase, InteractionBaseCompiled


@jitclass(
    [
        ("_num_feat", int32),  # a scalar
        ("_chis", float64[:, ::1]),  # a C-continuous array
        ("_incomp_coef", float64),  # a scalar
    ]
)
class FloryHugginsInteractionCompiled(InteractionBaseCompiled):
    r"""Compiled class for Flory-Huggins interaction energy.

    Flory-Huggins interaction is the second-ordered interaction, whose energy reads,

    .. math::
        f(\{\phi_r\}) = \sum_{r,s=1}^{N_\mathrm{S}} \frac{\chi_{rs}}{2} \phi_r\phi_s .

    Note that here we use describe the system by features.

    """

    def __init__(self, chis: np.ndarray, chis_shift: float):
        r"""
        Args:
            chis:
                2D array with the size of :math:`N_\mathrm{S} \times N_\mathrm{S}`,
                containing the Flory-Huggins interaction matrix :math:`\chi_{rs}` for
                features. The number of features :math:`N_\mathrm{S}` is inferred from
                this matrix.
            chis_shift:
                The shift of entire Flory-Huggins matrix for the :meth:`incomp_coef`.
        """
        self._num_feat = chis.shape[0]
        self._chis = chis  # do not affect chis
        self._incomp_coef = chis.sum() + chis_shift * self._num_feat * self._num_feat

    @property
    def num_feat(self) -> int:
        return self._num_feat

    def volume_derivative(
        self, potential: np.ndarray, phis_feat: np.ndarray
    ) -> np.ndarray:
        # since Flory-Huggins free energy contains only 2nd-ordered interactions,
        # the interaction energy is directly calculated from potential and phis
        ans = np.zeros_like(potential[0])
        for itr in range(self._num_feat):
            ans += potential[itr] * phis_feat[itr]
        ans *= 0.5
        return ans

    def potential(self, phis_feat: np.ndarray) -> np.ndarray:
        return self._chis @ phis_feat

    def incomp_coef(self, phis_feat: np.ndarray) -> float:
        return self._incomp_coef


class FloryHugginsInteraction(InteractionBase):
    r"""Class for Flory-Huggins interaction energy of mixture.

    The particular form of interaction energy density reads

        .. math::
            f_\mathrm{interaction}(\{\phi_i\}) = \sum_{i,j=1}^{N_\mathrm{C}} \frac{\chi_{ij}}{2} \phi_i\phi_j

        where :math:`\phi_i` is the fraction of component :math:`i`.
    """

    def __init__(
        self,
        num_comp: int,
        chis: np.ndarray | float,
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
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
        self._chis = 0.5 * (chis + chis.T)

    @property
    def chis(self) -> np.ndarray:
        r"""The Flory-Huggins interaction matrix of components :math:`\chi_{ij}`."""
        return self._chis

    @chis.setter
    def chis(self, chis_new: np.ndarray):
        chis_new = np.atleast_1d(chis_new)
        shape = (self.num_comp, self.num_comp)
        chis_new = np.array(np.broadcast_to(chis_new, shape))
        if not np.allclose(chis_new, chis_new.T):
            self._logger.warning("Using symmetrized χ interaction-matrix")
        self._chis = 0.5 * (chis_new + chis_new.T)

    @property
    def independent_entries(self) -> np.ndarray:
        r"""Entries of the upper triangle of the :math:`\chi_{ij}`"""

        return self._chis[np.triu_indices_from(self._chis, k=0)]

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
                Number of components :math:`N_\mathrm{C}`.
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
        rng: np.random.Generator | None = None,
    ):
        r"""Create Flory-Huggins interaction with random :math:`\chi_{ij}` matrix.

        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
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
        self._chis = np.full((self.num_comp, self.num_comp), chi)
        if vanishing_diagonal:
            self._chis[np.diag_indices_from(self._chis)] = 0

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

        self._chis[:] = 0  # reset old values

        # determine random entries
        if vanishing_diagonal:
            num_entries = self.num_comp * (self.num_comp - 1) // 2
        else:
            num_entries = self.num_comp * (self.num_comp + 1) // 2
        chi_vals = rng.normal(chi_mean, chi_std, num_entries)

        # build symmetric  matrix from this
        i, j = np.triu_indices(self.num_comp, 1 if vanishing_diagonal else 0)
        self._chis[i, j] = chi_vals
        self._chis[j, i] = chi_vals

    def _compiled_impl(
        self, *, additional_chis_shift: float = 1.0
    ) -> FloryHugginsInteractionCompiled:
        """Implementation of creating a compiled interaction instance.

        This method overwrites the interface
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
            self._chis, -self._chis.min() + additional_chis_shift
        )

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating interaction energy.

        This method overwrites the interface
        :meth:`~flory.entropy.base.InteractionBase._energy_impl` in
        :class:`~flory.entropy.base.InteractionBase`.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The interaction energy density
        """
        ans = 0.5 * np.einsum("...i,...j,ij->...", phis, phis, self._chis)
        return ans

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Jacobian :math:`\partial f_\mathrm{interaction}/\partial \phi_i`.

        This method overwrites the interface
        :meth:`~flory.entropy.base.InteractionBase._jacobian_impl` in
        :class:`~flory.entropy.base.InteractionBase`.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The full Jacobian
        """
        ans = phis @ self._chis
        return ans

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Hessian :math:`\partial^2 f_\mathrm{interaction}/\partial \phi_i^2`.

        This method overwrites the interface
        :meth:`~flory.entropy.base.InteractionBase._hessian_impl` in
        :class:`~flory.entropy.base.InteractionBase`.

        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.

        Returns:
            : The full Hessian
        """
        return np.zeros_like(phis)[..., None] + self._chis  # type: ignore
