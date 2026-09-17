"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

import numpy as np

import flory


def test_canonical_ensemble_binary_ideal():
    """test ReactiveCanonicalEnsemble in binary ideal limit"""
    num_comp = 2
    chis = [[0, 0], [0, 0]]
    weights = [[1, 1]]  # sum of species is conserved
    constraint = [1]  # sum of species is equal to one
    free_energy = flory.FloryHuggins(num_comp, chis)

    # use canonical-reaction ensemble
    ensemble = flory.ReactiveCanonicalEnsemble(num_comp, weights, constraint)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, ensemble
    )
    phases = finder.run().get_clusters().sort().normalize()

    assert phases.num_phases == 1
    np.testing.assert_allclose(phases.volumes, [1], rtol=1e-3)
    np.testing.assert_allclose(phases.fractions, [[0.5, 0.5]], rtol=1e-3)


def test_canonical_ensemble_binary_nonideal():
    """test ReactiveCanonicalEnsemble in binary case"""
    num_comp = 2
    chis = [[0, 3], [3, 0]]
    weights = [[1, 1]]  # sum of species is conserved
    constraint = [1]  # sum of species is equal to one
    free_energy = flory.FloryHuggins(num_comp, chis)

    # use canonical-reaction ensemble
    ensemble = flory.ReactiveCanonicalEnsemble(num_comp, weights, constraint)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, ensemble
    )
    phases = finder.run().get_clusters().sort().normalize()

    assert phases.num_phases == 2
    assert phases.volumes.sum() == pytest.approx(1)
    mus = free_energy.chemical_potentials(phases.fractions)
    np.testing.assert_allclose(mus, mus.mean(), rtol=1e-3)


def test_canonical_ensemble_limiting_case():
    """test ReactiveCanonicalEnsemble in limiting case"""
    num_comp = 3
    chis = [[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
    phi_means = [0.16, 0.68, 0.16]
    weights = np.eye(3)
    free_energy = flory.FloryHuggins(num_comp, chis)

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


def test_canonical_ensemble_3comp_2react():
    """test ReactiveCanonicalEnsemble in case of 3 components and 2 reactions"""
    num_comp = 3
    chis = [[0, 3.5, 2.5], [3.5, 0, 3], [2.5, 3, 0]]
    weights = [[1, 0, 0], [0, 1, 1]]
    constraint = [0.33, 1 - 0.33]
    free_energy = flory.FloryHuggins(num_comp, chis)

    # use canonical-reaction ensemble
    ensemble = flory.ReactiveCanonicalEnsemble(num_comp, weights, constraint)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, ensemble
    )
    phases = finder.run().get_clusters().sort().normalize()

    assert phases.num_phases == 2
    assert phases.mean_fractions[0] == pytest.approx(0.33)
    assert phases.mean_fractions[1:].sum() == pytest.approx(0.67)

    mus = free_energy.chemical_potentials(phases.fractions)
    mu_conserv = mus[:, 0].mean()
    mu_react = mus[:, 1:].mean()
    assert mu_conserv != pytest.approx(mu_react)
    np.testing.assert_allclose(mus[:, 0], mu_conserv, rtol=1e-3)
    np.testing.assert_allclose(mus[:, 1:], mu_react, rtol=1e-3)
