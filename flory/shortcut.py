"""Shortcuts for simple tasks of finding coexisting phases."""

import numpy as np
from typing import Optional

from .free_energy import FloryHuggins
from .ensemble import CanonicalEnsemble
from .mcmp import CoexistingPhasesFinder


def find_coexisting_phases(
    num_comp: int,
    chis: np.ndarray,
    phi_means: np.ndarray,
    sizes: Optional[np.ndarray] = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Find coexisting phases of Flory-Huggins mixtures in canonical ensemble.

    This function is a convenience wrapper for the class
    :class:`~flory.mcmp.finder.CoexistingPhasesFinder`. This function will create the
    class :class:`~flory.mcmp.finder.CoexistingPhasesFinder` internally, conduct the
    random initialization, and then use self consistent iterations to find coexisting
    phases. See class :class:`~flory.mcmp.finder.CoexistingPhasesFinder` for more details
    on the supported arguments.

    Args:
        chis:
            The interaction matrix. Symmetric 2D array with size of :math:`N_\mathrm{c}
            \times N_\mathrm{c}`. This matrix should be the full :math:`\chi_{ij}` matrix
            of the system, including the solvent component.
        phi_means:
            The average volume fractions :math:`\bar{\phi}_i` of all the components of the
            system. 1D array of length :math:`N_\mathrm{c}`. Note that the volume fraction
            of the solvent is included as well, so the sum of this array must be one.
        num_part:
            Number of compartments :math:`M` in the system.
        \**kwargs:
            All additional arguments are used directly to initialize
            :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.

    Returns:
        [0]:
            Volume fractions of each phase :math:`J_p`. 1D array with the size of
            :math:`N_\mathrm{p}`.
        [1]:
            Volume fractions of components in each phase :math:`\phi_{p,i}`. 2D
            array with the size of :math:`N_\mathrm{p} \times N_\mathrm{c}`.
    """
    free_energy = FloryHuggins(num_comp, chis, sizes)
    ensemble = CanonicalEnsemble(num_comp, phi_means)
    finder = CoexistingPhasesFinder(
        free_energy.interaction,
        free_energy.entropy,
        ensemble,
        **kwargs,
    )
    return finder.run()
