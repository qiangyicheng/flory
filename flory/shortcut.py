"""Shortcuts for simple tasks of finding coexisting phases.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import linalg

from .common import PhasesResult
from .constraint import ConstraintBase, LinearLocalConstraint, NoConstraint
from .ensemble import CanonicalEnsemble, EnsembleBase, SemiGrandCanonicalEnsemble
from .free_energy import FloryHuggins
from .mcmp import CoexistingPhasesFinder

_logger = logging.getLogger(__name__)


def prepare_reactions(
    stoichiometric_matrix: np.ndarray,
    phi_means: np.ndarray,
    *,
    mus: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
) -> tuple[EnsembleBase, ConstraintBase]:
    """Prepare an ensemble and conservation constraint for a reaction system.

    Components that participate in at least one reaction are treated grand
    canonically, with chemical potentials and relative molecule sizes given by ``mus``
    and ``sizes``. Components absent from all reactions remain canonical. The returned
    local constraint enforces the conserved quantities implied by the stoichiometric
    matrix, initialized using ``phi_means``.

    Args:
        stoichiometric_matrix:
            Matrix with one reaction per row and one component per column.
        phi_means:
            Mean volume fractions of all components.
        mus:
            Chemical potentials for the components, defaulting to zero for all
            components.
        sizes:
            Relative molecule volumes for the components, defaulting to one for
            all components.

    Returns:
        The ensemble and the corresponding constraint for the reaction system.
    """
    # TODO add tests

    # parse the stoichiometric_matrix
    stoichiometric_matrix = np.atleast_2d(stoichiometric_matrix)
    _, num_comp = stoichiometric_matrix.shape
    if mus is None:
        mus = np.zeros(num_comp)
    else:
        mus = np.atleast_1d(mus)
    if sizes is None:
        sizes = np.ones(num_comp)
    else:
        sizes = np.atleast_1d(sizes)

    # check arguments for consistency
    if not np.allclose(stoichiometric_matrix.sum(axis=1), 0):
        _logger.warning("Stoichiometric matrix violates volume conservation.")

    # determine the reacting components
    reacting = np.any(stoichiometric_matrix != 0, axis=0)

    if not np.any(reacting):
        return CanonicalEnsemble(num_comp, phi_means), NoConstraint(num_comp)

    # determine the ensemble
    ensemble = SemiGrandCanonicalEnsemble(
        num_comp, is_canonical=~reacting, constraint=phi_means
    )
    for i in np.flatnonzero(reacting):
        ensemble.set_chemical_potential(i, mus[i], sizes[i])

    # determine the constraint based on the cokernel of stoichiometric matrix
    b = linalg.null_space(stoichiometric_matrix[:, reacting])
    constraint_matrix = np.zeros((b.shape[1], num_comp))
    constraint_matrix[:, reacting] = b.T
    constraint_value = constraint_matrix @ phi_means
    constraint = LinearLocalConstraint(num_comp, constraint_matrix, constraint_value)

    return ensemble, constraint


def find_coexisting_phases(
    num_comp: int,
    chis: np.ndarray,
    phi_means: np.ndarray,
    sizes: np.ndarray | None = None,
    **kwargs,
) -> PhasesResult:
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
        **kwargs:
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
    # cluster (using default distance threshold), sort, and normalize the phases
    phases = phases.get_clusters().sort().normalize()
    # return result together with diagnostic information
    return PhasesResult.from_phases(phases, info=finder.diagnostics.copy())
