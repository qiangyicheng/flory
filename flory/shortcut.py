"""Shortcuts for simple tasks of finding coexisting phases."""
from __future__ import annotations

import numpy as np

from .common.phases import Phases
from .ensemble import CanonicalEnsemble
from .free_energy import FloryHuggins
from .mcmp import CoexistingPhasesFinder


def find_coexisting_phases(
    num_comp: int,
    chis: np.ndarray,
    phi_means: np.ndarray,
    sizes: np.ndarray | None = None,
    **kwargs,
) -> Phases:
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
        phases:
            Composition and relative size of the phases. The first item (accessible by
            :code:`phases[0]` or :code:`phases.Js`) contains the fraction of volume of
            each phase. The second item (accessible by :code:`phases[1]` or
            :code:`phases.phis`) contains volume fractions of all components.
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
