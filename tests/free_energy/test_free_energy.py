"""
.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np
import pytest

import flory
from flory.common.phases import get_uniform_random_composition


def test_equilibration_error():
    """Test function `equilibration_error`"""
    num_comp = 3
    chis = np.array([[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]])
    phi_means = np.array([0.16, 0.55, 0.29])
    sizes = np.array([2.0, 2.0, 1.0])

    fh = flory.FloryHuggins(num_comp, chis, sizes)
    ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
    finder = flory.CoexistingPhasesFinder(fh.interaction, fh.entropy, ensemble)

    phases = finder.run().get_clusters()

    phase_error = fh.equilibration_error(phases.fractions)
    assert phase_error.max() < 1e-4

    random_comps = [get_uniform_random_composition(3) for _ in range(3)]
    phase_error = fh.equilibration_error(random_comps)
    assert phase_error.max() > 1e-4
