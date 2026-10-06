"""
.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np
import pytest
from helpers import assert_phases_allclose

import flory


@pytest.mark.slow
@pytest.mark.no_cover
def test_consistency_ensemble():
    num_comp = 3
    chis = [[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
    phi_means = [0.16, 0.68, 0.16]
    sizes = [2.0, 2.0, 1.0]

    free_energy = flory.FloryHuggins(num_comp, chis, sizes)
    ensemble = flory.CanonicalEnsemble(num_comp, phi_means)

    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, ensemble
    )

    phases_canonical = finder.run().get_clusters().sort()

    ###############################################################

    mus = [0.0, 0.0, 0.0]
    ensemble = flory.GrandCanonicalEnsemble.from_chemical_potential(
        num_comp, mus, sizes
    )
    constraint = flory.LinearGlobalConstraint(
        num_comp, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [phi_means[0], phi_means[1]]
    )
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction,
        free_energy.entropy,
        ensemble,
        constraint,
        random_std=1.0,
    )

    phases_grandcanonical = finder.run().get_clusters().sort()

    np.testing.assert_allclose(
        phases_canonical.volumes, phases_grandcanonical.volumes, rtol=1e-3
    )
    np.testing.assert_allclose(
        phases_canonical.fractions, phases_grandcanonical.fractions, rtol=1e-3
    )


@pytest.mark.slow
@pytest.mark.no_cover
def test_consistency_polydispersity():
    num_feat = 2
    chis_feat = [[0, 4.0], [4.0, 0]]
    phi_means = [0.2, 0.3, 0.2, 0.3]
    sizes = [1, 2, 1, 2]
    num_comp_per_feat = [2, 2]
    num_comp = np.sum(num_comp_per_feat)

    interaction = flory.FloryHugginsBlockInteraction(
        num_feat, chis_feat, num_comp_per_feat
    )
    entropy = flory.IdealGasPolydispersedEntropy(num_feat, sizes, num_comp_per_feat)
    ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
    finder = flory.CoexistingPhasesFinder(interaction, entropy, ensemble)

    phases_optimized = finder.run().get_clusters().sort()
    ###############################################################

    chis = interaction.chis
    print(chis)
    fh = flory.FloryHuggins(num_comp, chis, sizes)
    finder = flory.CoexistingPhasesFinder(fh.interaction, fh.entropy, ensemble)

    phases_standard = finder.run().get_clusters().sort()
    assert_phases_allclose(phases_optimized, phases_standard, tol=1e-3)


def test_semi_grandcanonical_canonical_ensemble():
    num_comp = 3
    chis = [[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
    phi_means = [0.16, 0.68, 0.16]
    sizes = [2.0, 2.0, 1.0]

    free_energy = flory.FloryHuggins(num_comp, chis, sizes)

    canonical = flory.CanonicalEnsemble(num_comp, phi_means)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, canonical
    )
    phases_canonical = finder.run().get_clusters().sort()

    semi = flory.SemiGrandCanonicalEnsemble(
        num_comp, is_canonical=True, constraint=phi_means
    )
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, semi
    )
    phases_semi = finder.run().get_clusters().sort()

    assert_phases_allclose(phases_canonical, phases_semi, tol=1e-3)


def test_semi_grandcanonical_grandcanonical_ensemble():
    num_comp = 3
    chis = [[0, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
    mus = [0.16, 0.68, 0.16]
    sizes = [2.0, 2.0, 1.0]

    free_energy = flory.FloryHuggins(num_comp, chis, sizes)

    grandcanonical = flory.GrandCanonicalEnsemble.from_chemical_potential(
        num_comp, mus, sizes
    )
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, grandcanonical
    )
    phases_grandcanonical = finder.run().get_clusters().sort()

    semi = flory.SemiGrandCanonicalEnsemble(
        num_comp, is_canonical=False, constraint=grandcanonical.scaled_activity
    )
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, semi
    )
    phases_semi = finder.run().get_clusters().sort()
    assert_phases_allclose(phases_grandcanonical, phases_semi, tol=1e-3)
