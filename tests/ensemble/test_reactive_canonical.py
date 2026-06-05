"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

import flory


def test_canonical_ensemble_binary():
    """test ReactiveCanonicalEnsemble in binary simple limit"""

    num_comp = 2
    chis = [[0, 3], [3, 0]]
    sizes = [1.0, 1.0]
    weights = [[1, 1]]
    constraint = [1]

    free_energy = flory.FloryHuggins(num_comp, chis, sizes)

    # use canonical-reaction ensemble
    ensemble = flory.ReactiveCanonicalEnsemble(num_comp, weights, constraint)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, ensemble
    )
    phases = finder.run().get_clusters().sort()

    assert phases.num_phases == 1
    np.testing.assert_allclose(phases.volumes, [1], rtol=1e-3)
    np.testing.assert_allclose(phases.fractions, [[0.5, 0.5]], rtol=1e-3)


def test_canonical_ensemble_limiting_case():
    """test ReactiveCanonicalEnsemble in limiting case"""

    num_comp = 3
    chis = [[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
    phi_means = [0.16, 0.68, 0.16]
    sizes = [2.0, 2.0, 1.0]
    weights = np.eye(3)

    free_energy = flory.FloryHuggins(num_comp, chis, sizes)

    # use canonical ensemble
    ensemble1 = flory.CanonicalEnsemble(num_comp, phi_means)
    finder1 = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, ensemble1
    )
    phases1 = finder1.run().get_clusters().sort()

    # use canonical-reaction ensemble
    ensemble2 = flory.ReactiveCanonicalEnsemble(num_comp, weights, phi_means)
    finder2 = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, ensemble2
    )
    phases2 = finder2.run().get_clusters().sort()

    assert phases1.num_phases == phases2.num_phases
    np.testing.assert_allclose(phases1.volumes, phases2.volumes, rtol=1e-3)
    np.testing.assert_allclose(phases1.fractions, phases2.fractions, rtol=1e-3)
