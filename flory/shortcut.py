"""Shortcuts for simple tasks of finding coexisting phases.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""
from __future__ import annotations

import numpy as np

from .common import Phases
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
        num_comp:
            Number of components :math:`N_\mathrm{C}` in the system.
        chis:
            The interaction matrix. Symmetric 2D array with size of :math:`N_\mathrm{C}
            \times N_\mathrm{C}`. This matrix should be the full :math:`\chi_{ij}` matrix
            of the system, including the solvent component.
        phi_means:
            The average volume fractions :math:`\bar{\phi}_i` of all the components of the
            system. 1D array of length :math:`N_\mathrm{C}`. Note that the volume fraction
            of the solvent is included as well, so the sum of this array must be one.
        sizes:
            The relative molecule volumes :math:`l_i = \nu_i/\nu` with respect to the
            volume of a reference molecule :math:`\nu`. It is treated as all-one vector by
            default.
        \**kwargs:
            All additional arguments are used directly to initialize
            :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.

    Returns:
        :
            Composition and relative size of the phases. The member
            :paramref:`~flory.common.phases.Phases.volumes` (accessible by
            :code:`.volumes`) contains the fraction of volume of each phase. The member
            :paramref:`~flory.common.phases.Phases.fractions` (accessible by
            :code:`.fractions`) contains volume fractions of all components.
    """
    free_energy = FloryHuggins(num_comp, chis, sizes)
    ensemble = CanonicalEnsemble(num_comp, phi_means)
    finder = CoexistingPhasesFinder(
        free_energy.interaction,
        free_energy.entropy,
        ensemble,
        **kwargs,
    )
    phases = finder.run()
    return phases.get_clusters()  # use default distance threshold