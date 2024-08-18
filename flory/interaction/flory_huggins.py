"""
Module defining thermodynamic quantities of multicomponent phase separation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
from typing import Any, Union, Optional

from numba.experimental import jitclass
from numba import float64, int32
import numpy as np
from .base import InteractionBase


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_chis", float64[:, ::1]),  # a C-continuous array
        ("_incomp_coef", float64),  # a scalar
    ]
)
class FloryHugginsInteractionCompiled(object):
    def __init__(self, chis: np.ndarray, chis_shift: float):
        r"""The compiled FLory-Huggins interaction.

        Args:
            chis:
                The Flory-Huggins matrix :math:`\chi_{ij}`. The number of components is
                inferred from this matrix.
            chis_shift:
                The shift of entire Flory-Huggins matrix for the calculation.
        """
        self._num_comp = chis.shape[0]
        self._chis = chis  # do not affect chis
        self._incomp_coef = chis.sum() + chis_shift * self._num_comp * self._num_comp

    @property
    def num_comp(self):
        return self._num_comp

    @property
    def chis(self):
        return self._chis

    def energy(self, potential: np.ndarray, phis: np.ndarray) -> float:
        # since Flory-Huggins free energy contains only 2nd-ordered interactions,
        # the interaction energy is directly calculated from potential and phis
        ans = np.zeros_like(potential[0])
        for itr in range(self._num_comp):
            ans += potential[itr] * phis[itr]
        ans *= 0.5
        return ans

    def potential(self, phis: np.ndarray) -> np.ndarray:
        ans = np.zeros_like(phis)
        for itr_i in range(self._num_comp):
            for itr_j in range(self._num_comp):
                current_chi = self._chis[itr_i][itr_j]
                if current_chi == 0.0:
                    continue
                ans[itr_i] += current_chi * phis[itr_j]

        return ans  # same as self._chis @ phis

    def incomp_coef(self, phis: np.ndarray) -> float:
        return self._incomp_coef


class FloryHugginsInteraction(InteractionBase):
    r"""represents the interaction energy of a Flory-Huggins multicomponent mixture

    The particular implementation of the interaction energy density reads

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
        """
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`. In the cases with degenerate
                components, this value should be interpreted as number of features
                :math:`N_\mathrm{f}`.
            chis:
                The Flory-Huggins interaction matrix
        """
        super().__init__(num_comp=num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)

        chis = np.atleast_1d(chis)

        shape = (num_comp, num_comp)
        chis = np.array(np.broadcast_to(chis, shape))

        # ensure that the chi matrix is symmetric
        if not np.allclose(chis, chis.T):
            logging.warning("Using symmetrized χ interaction-matrix")
        self.chis = 0.5 * (chis + chis.T)

    @property
    def independent_entries(self) -> np.ndarray:
        r"""entries of the upper triangle of the :math:`\chi_{ij}` matrix"""
        return self.chis[np.triu_indices_from(self.chis, k=0)]

    @classmethod
    def from_uniform(
        cls,
        num_comp: int,
        chi: float,
        *,
        vanishing_diagonal: bool = True,
    ):
        r"""create Flory-Huggins free energy with uniform `chis` matrix

        Args:
            num_comp:
                The number of components
            chi:
                The value of all non-zero values in the interaction matrix :math:`\chi{i \ne j}`
            vanishing_diagonal:
                Whether the diagonal elements of the `chis` matrix are set to be zero.
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
        r"""create Flory-Huggins free energy with random `chis` matrix

        Args:
            num_comp:
                Number of components
            chi_mean:
                Mean interaction
            chi_std:
                Standard deviation of the interactions
            vanishing_diagonal:
                Whether the diagonal elements of the `chis` matrix are set to be zero.
            rng:
                the random number generator

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
        r"""choose random interaction parameters

        Args:
            chi:
                The value of all non-zero values in the interaction matrix :math:`\chi{i \ne j}`
            vanishing_diagonal:
                Whether the diagonal elements of the `chis` matrix are set to be zero.
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
        """choose random interaction parameters

        Args:
            chi_mean:
                Mean interaction
            chi_std:
                Standard deviation of the interactions
            vanishing_diagonal:
                Whether the diagonal elements of the `chis` matrix are set to be zero.
            rng:
                the random number generator
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
        """make the interaction instance that can be used by the :mod:`mcmp` module.

        Args:
            additional_chis_shift:
                Shift of the entire chis matrix to improve the convergence by evolving
                towards incompressible system faster. This value should be larger than 0.
                This value only affects the numerics, not the actual physical system. Note
                that with very large value, the convergence will be slowed down, since the
                algorithm no longer have enough ability to temporarily relax the
                incompressibility.
        Returns:
            : An instance of :class:`FloryHugginsInteractionCompiled`.
        """

        return FloryHugginsInteractionCompiled(
            self.chis, -self.chis.min() + additional_chis_shift
        )

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        """returns interaction energy density for a given composition

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
        r"""returns full Jacobian :math:`\partial f/\partial \phi` for the given composition

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
        """returns Hessian :math:`\partial^2 f/\partial \phi^2` for the given composition

        Args:
            phis:
                The composition of the phase(s). if multiple phases are included, the
                index of the components must be the last dimension.
        Returns:
            : The full Hessian
        """
        return np.zeros_like(phis)[..., None] + self.chis  # type: ignore
