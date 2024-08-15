"""
Module defining thermodynamic quantities of multicomponent phase separation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from typing import Callable

import numba as nb
import numpy as np
import scipy.linalg
from scipy.cluster import hierarchy
from scipy.spatial import distance
from .base import FreeEnergyBase


class SolventFractionError(RuntimeError):
    """error indicating that the solvent fraction was not in [0, 1]"""

    pass


class FloryHuggins(FreeEnergyBase):
    r"""represents the free energy of a multicomponent mixture

    The particular implementation of the free energy density reads

    .. math::
        f(\{\phi_i\}) = \frac{k_\mathrm{B}T}{\nu}\biggl[
            \phi_0\ln(\phi_0)
            + \sum_{i=1}^N \frac{\nu}{\nu_i}\phi_i \ln(\phi_i)
            + \!\sum_{i,j=1}^N \frac{\chi_{ij}}{2} \phi_i\phi_j
        \biggr]

    where :math:`\phi_i` is the fraction of component :math:`i` and :math:`\phi_0 = 1 -
    \sum_i \phi_i` is the fraction of the solvent. All components are assumed to have the
    same molecular volume :math:`\nu` by default and the interactions are quantified by
    the Flory matrix :math:`\chi_{ij}`. Note that components do not interact with the
    solvent, which is thus completely inert. The relative molecular volume :math:`v_i` can
    be changed by setting the optional parameter `sizes`, which always treats the volume
    of the solvent molecular as unit.
    """

    def __init__(
        self,num_comp: int, 
        chis: np.ndarray,
        size: np.ndarray | None = None,
    ):
        """
        Args:
            chis (:class:`~numpy.ndarray`):
                The interaction matrix
            sizes (:class:`~numpy.ndarray | None`):
                The relative volumes with respect to the volume of the solvent molecular.
                It is treated as all-one vector by default or passing `None`.
        """
        super().__init__(num_comp=num_comp, size=size)
        chis = np.atleast_1d(chis)

        shape = (num_comp, num_comp)
        chis = np.array(np.broadcast_to(chis, shape))

        # ensure that the chi matrix is symmetric
        if not np.allclose(chis, chis.T):
            logging.warning("Using symmetrized χ interaction-matrix")
        self.chis = 0.5 * (chis + chis.T)


    def make_interaction(self):
        @jitclass
        ...

    @property
    def independent_entries(self) -> np.ndarray:
        """:class:`~numpy.ndarray` entries of the upper triangle only"""
        return self.chis[np.triu_indices_from(self.chis, k=0)]  # type: ignore


    @classmethod
    def from_uniform(
        cls,
        num_comp: int,
        chi: float,
        *,
        size: np.ndarray | None = None,
        vanishing_diagonal:bool=True
    ) -> FloryHuggins:
        """create Flory-Huggins free energy with uniform chi matrix

        Args:
            num_comp (int):
                The number of components
            chi (float):
                The value of all non-zero values in the interaction matrix
            inert_solvent (bool):
                Flag determining whether the solvent (species 0) is assumed inert or not.
                For an inert solvent, the diagonal of the `chi` matrix must vanish.
            sizes (:class:`~numpy.ndarray | None`):
                The relative volumes with respect to the volume of the solvent molecular.
                It is treated as all-one vector by default or passing `None`.
        """
        chis = np.full((num_comp, num_comp), chi)
        if vanishing_diagonal:
            chis[np.diag_indices_from(chis)] = 0
        return cls(num_comp, chis, size=size)

    @classmethod
    def from_random_normal(
        cls,
        num_comp: int,
        chi_mean: float = 0,
        chi_std: float = 1,
        *,
        vanishing_diagonal: bool = True,
        size: np.ndarray | None = None,
        rng=None,
    ) -> FloryHuggins:
        """create random Flory-Huggins free energy density

        Args:
            num_comp (int):
                Number of components (excluding the solvent)
            chi_mean (float):
                Mean interaction
            chi_std (float):
                Standard deviation of the interactions
            inert_solvent (bool):
                Flag determining whether the solvent (species 0) is assumed inert or not.
                For an inert solvent, the diagonal of the `chi` matrix must vanish.
            rng:
                the random number generator
            sizes (:class:`~numpy.ndarray | None`):
                The relative volumes with respect to the volume of the solvent molecular.
                It is treated as all-one vector by default or passing `None`.
        """
        obj = cls(num_comp, 
            np.zeros((num_comp, num_comp)), size=size
        )
        obj.set_random_chis(chi_mean, chi_std, vanishing_diagonal=vanishing_diagonal,rng=rng)
        return obj


    def set_random_chis(self, chi_mean: float = 0, chi_std: float = 1, *, vanishing_diagonal: bool = True,rng=None):
        """choose random interaction parameters

        Args:
            chi_mean: Mean interaction
            chi_std: Standard deviation of the interactions
            rng: the random number generator
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


    def free_energy_density(
        self, phis: np.ndarray, *, check: bool = True
    ) -> np.ndarray:
        """returns free energy for a given composition

        Args:
            phis (:class:`numpy.ndarray`): The composition of the phase(s)
            check (bool): Whether the solvent fraction is checked to be positive
        """
        phis = np.asanyarray(phis)
        assert phis.shape[-1] == self.num_comp, "Wrong component count"

        phi_sol = 1 - phis.sum(axis=-1)
        if check and np.any(phi_sol < 0):
            raise SolventFractionError("Solvent has negative concentration")

        entropy_comp = np.einsum("...i,...i->...", phis / self.size, np.log(phis))
        entropy_sol = phi_sol * np.log(phi_sol)
        enthalpy = 0.5 * np.einsum("...i,...j,ij->...", phis, phis, self.chis)
        return entropy_comp + entropy_sol + enthalpy  # type: ignore

    def chemical_potentials(self, phis: np.ndarray) -> np.ndarray:
        """returns chemical potentials for a given composition"""
        phis = np.asanyarray(phis)
        phi_sol = 1 - phis.sum(axis=-1, keepdims=True)
        if np.any(phi_sol < 0):
            raise SolventFractionError("Solvent has negative concentration")
        return (  # type: ignore
            np.log(phis) / self.size
            - np.log(phi_sol)
            + (1.0 / self.size - 1.0)
            + np.einsum("...i,ij->...j", phis, self.chis)
        )

    def hessian(self, phis: np.ndarray) -> np.ndarray:
        """returns Hessian for the given composition"""
        phis = np.asanyarray(phis)
        assert phis.shape == (self.num_comp,)
        phi_sol = 1 - phis.sum(axis=-1, keepdims=True)
        if np.any(phi_sol < 0):
            raise SolventFractionError("Solvent has negative concentration")
        return np.eye(self.num_comp) / (phis * self.sizes) + 1 / phi_sol + self.chis  # type: ignore
