import numpy as np

from free_energy.flory_huggins import FloryHuggins
from mcmp.mcmp import CoexistingPhasesFinder


def find_coexisting_phases(
    num_comp: int,
    chis: np.ndarray,
    phi_means: np.ndarray,
    size: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Find coexisting phases of a Flory-Huggins mixtures.

    This function is a convenience wrapper for the class :class:`CoexistingPhasesFinder`.
    This function will create the class :class:`CoexistingPhasesFinder` internally,
    conduct the random initialization, and then use self consistent iterations to
    find coexisting phases. See class :class:`CoexistingPhasesFinder` for more details
    on the supported arguments.

    Args:
        chis:
            The interaction matrix. Symmetric 2D array with size of :math:`N_\mathrm{c}
            \times N_\mathrm{c}`. This matrix should be the full :math:`\chi_{ij}`
            matrix of the system, including the solvent component.
        phi_means:
            The average volume fractions :math:`\bar{\phi}_i` of all the components of
            the system. 1D array of length :math:`N_\mathrm{c}`. Note that the volume
            fraction of the solvent is included as well, so the sum of this array must
            be one.
        num_part:
            Number of compartments :math:`M` in the system.
        \**kwargs:
            All additional arguments are used directly to initialize
            :class:`CoexistingPhasesFinder`.

    Returns:
        [0]:
            Volume fractions of each phase :math:`J_\alpha`. 1D array with the size of
            :math:`N_\mathrm{p}`.
        [1]:
            Volume fractions of components in each phase :math:`\phi_i^{(\alpha)}`. 2D
            array with the size of :math:`N_\mathrm{p} \times N_\mathrm{c}`.
    """
    free_energy = FloryHuggins(num_comp, chis, size)
    finder = CoexistingPhasesFinder(free_energy, phi_means, **kwargs)
    return finder.run()
