"""
Module defining thermodynamic quantities of multicomponent phase separation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Union, Optional

import numpy as np

from ..interaction import FloryHugginsInteraction
from ..entropy import IdealGasEntropy
from .base import FreeEnergyBase


class FloryHuggins(FreeEnergyBase):
    r"""represents the free energy of a multicomponent mixture

    The particular implementation of the free energy density reads

    .. math::
        f(\{\phi_i\}) = \frac{k_\mathrm{B}T}{\nu}\biggl[
            \sum_{i=1}^N \frac{\nu}{\nu_i}\phi_i \ln(\phi_i)
            + \!\sum_{i,j=1}^{N_\mathrm{c}} \frac{\chi_{ij}}{2} \phi_i\phi_j
        \biggr]

    where :math:`\phi_i` is the fraction of component :math:`i`. All components are
    assumed to have the same molecular volume :math:`\nu` by default and the interactions
    are quantified by the Flory matrix :math:`\chi_{ij}`. The relative molecular sizes
    :math:`l_i=\nu_i/\nu` can be changed by setting the optional parameter `sizes`. Note
    that no implicit solvent is assumed.
    """

    def __init__(
        self,
        num_comp: int,
        chis: Union[np.ndarray, float],
        sizes: Optional[np.ndarray] = None,
    ):
        """
        Args:
            num_comp:
                Number of components in the system
            chis:
                The Flory-Huggins interaction matrix
            sizes:
                The relative volumes with respect to the volume of an imaginary reference
                molecular. It is treated as all-one vector by default or passing `None`.
        """
        interaction = FloryHugginsInteraction(num_comp, chis=chis)
        entropy = IdealGasEntropy(num_comp, sizes=sizes)
        super().__init__(interaction, entropy)

    @classmethod
    def from_uniform(
        cls,
        num_comp: int,
        chi: float,
        *,
        sizes: Optional[np.ndarray] = None,
        vanishing_diagonal: bool = True,
    ):
        r"""create Flory-Huggins free energy with uniform `chis` matrix

        Args:
            num_comp:
                The number of components
            chi:
                The value of all non-zero values in the interaction matrix :math:`\chi{i \ne j}`
            sizes:
                The relative volumes with respect to the volume of an imaginary reference
                molecular. It is treated as all-one vector by default or passing `None`.
            vanishing_diagonal:
                Whether the diagonal elements of the `chis` matrix are set to be zero.
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
        sizes: Optional[np.ndarray] = None,
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
            sizes:
                The relative volumes with respect to the volume of an imaginary reference
                molecular. It is treated as all-one vector by default or passing `None`.
            vanishing_diagonal:
                Whether the diagonal elements of the `chis` matrix are set to be zero.
            rng:
                the random number generator

        """
        obj = cls(num_comp, 0, sizes=sizes)
        obj.interaction.set_random_chis(
            chi_mean, chi_std, vanishing_diagonal=vanishing_diagonal, rng=rng
        )
        return obj
