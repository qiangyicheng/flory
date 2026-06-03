"""Module for Flory-Huggins free energy.
Flory-Huggins free energy is a combination of Flory-Huggins interaction energy and ideal
gas entropy.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np

from ..entropy import IdealGasEntropy
from ..interaction import FloryHugginsInteraction
from .base import FreeEnergyBase


class FloryHuggins(FreeEnergyBase):
    r"""represents the free energy of a multicomponent mixture

    The particular implementation of the free energy density reads

    .. math::
        f(\{\phi_i\}) =
            \sum_{i=1}^{N_\mathrm{C}} \frac{\nu}{\nu_i}\phi_i \ln(\phi_i)
            + \sum_{i,j=1}^{N_\mathrm{C}} \frac{\chi_{ij}}{2} \phi_i\phi_j

    where :math:`\phi_i` is the fraction of component :math:`i`. All components are
    assumed to have the same molecular volume :math:`\nu` by default and the interactions
    are quantified by the Flory matrix :math:`\chi_{ij}`. The relative molecular sizes
    :math:`l_i=\nu_i/\nu` can be changed by setting the optional parameter `sizes`. Note
    that no implicit solvent is assumed.
    """

    def __init__(
        self,
        num_comp: int,
        chis: np.ndarray | float,
        sizes: np.ndarray | None = None,
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            chis:
                The Flory-Huggins interaction matrix of components :math:`\chi_{ij}`.
            sizes:
                The relative molecule volumes :math:`l_i = \nu_i/\nu` with respect to the
                volume of a reference molecule :math:`\nu`. It is treated as all-one
                vector by default.
        """
        interaction = FloryHugginsInteraction(num_comp, chis=chis)
        entropy = IdealGasEntropy(num_comp, sizes=sizes)
        super().__init__(interaction, entropy)

    @property
    def chis(self) -> np.ndarray:
        r"""The Flory-Huggins interaction matrix of components :math:`\chi_{ij}`."""
        return self.interaction.chis

    @chis.setter
    def chis(self, chis_new: np.ndarray):
        r"""Set Flory-Huggins interaction matrix.

        Args:
            chis_new:
                Updated interaction matrix :math:`\chi_{ij}`.
        """
        self.interaction.chis = chis_new

    @property
    def sizes(self) -> np.ndarray:
        r"""The relative molecule volumes :math:`l_i = \nu_i/\nu`."""
        return self.entropy.sizes

    @sizes.setter
    def sizes(self, sizes_new: np.ndarray):
        r"""Set relative molecule volumes.

        Args:
            sizes_new:
                Updated relative molecule volumes :math:`l_i = \nu_i/\nu`.
        """
        self.entropy.sizes = sizes_new

    @classmethod
    def from_uniform(
        cls,
        num_comp: int,
        chi: float,
        *,
        sizes: np.ndarray | None = None,
        vanishing_diagonal: bool = True,
    ):
        r"""Create Flory-Huggins free energy with uniform :math:`\chi_{ij}` matrix.

        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            chi:
                Uniform off-diagonal interaction value.
            sizes:
                Relative molecule volumes :math:`l_i = \nu_i/\nu`.
            vanishing_diagonal:
                Whether diagonal entries of the interaction matrix are set to zero.

        Returns:
            Instance of :class:`FloryHuggins`.
        """
        obj = cls(num_comp, 0, sizes=sizes)
        obj.interaction.set_uniform_chis(chi, vanishing_diagonal=vanishing_diagonal)
        return obj

    @classmethod
    def from_random_normal(
        cls,
        num_comp: int,
        chi_mean: float = 0,
        chi_std: float = 1,
        *,
        sizes: np.ndarray | None = None,
        vanishing_diagonal: bool = True,
        rng: np.random.Generator | None = None,
    ):
        r"""Create Flory-Huggins free energy with random :math:`\chi_{ij}` matrix.

        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            chi_mean:
                Mean interaction :math:`\bar{\chi}`.
            chi_std:
                Standard deviation of interactions :math:`\sigma_\chi`.
            sizes:
                Relative molecule volumes :math:`l_i = \nu_i/\nu`.
            vanishing_diagonal:
                Whether diagonal entries of the interaction matrix are set to zero.
            rng:
                Random number generator used to sample interaction entries.

        Returns:
            Instance of :class:`FloryHuggins`.
        """
        obj = cls(num_comp, 0, sizes=sizes)
        obj.interaction.set_random_chis(
            chi_mean, chi_std, vanishing_diagonal=vanishing_diagonal, rng=rng
        )
        return obj
