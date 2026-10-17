"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging

import numpy as np
import pytest
from helpers import assert_phases_allclose

import flory


def test_reactions_1comp(caplog):
    num_comp = 1
    chis = [[0]]
    phi_means = [1.0]
    sizes = [2.0]

    ens, constr = flory.prepare_reactions([0], phi_means)
    free_energy = flory.FloryHuggins(num_comp, chis, sizes)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction,
        free_energy.entropy,
        ensemble=ens,
        constraints=constr,
        random_std=1.0,  # use less aggressive randomness to avoid rapid dying of compartments
    )

    phases = finder.run(progress=False).get_clusters()
    assert phases.num_phases == 1

    with caplog.at_level(logging.WARNING):
        flory.prepare_reactions([1], phi_means)
    assert "conservation" in caplog.text


def test_reactions_2comp(caplog):
    num_comp = 2
    chis = np.full((num_comp, num_comp), 0.5)
    phi_means = [0.4, 0.6]
    sizes = [2.0, 1.0]

    ens, constr = flory.prepare_reactions([1, -1], phi_means, sizes=sizes)
    free_energy = flory.FloryHuggins(num_comp, chis, sizes)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction,
        free_energy.entropy,
        ensemble=ens,
        constraints=constr,
        random_std=1.0,  # use less aggressive randomness to avoid rapid dying of compartments
    )

    phases = finder.run(progress=False).get_clusters()
    assert phases.num_phases == 1
    mus = free_energy.chemical_potentials(phases.fractions)
    assert mus.shape == (1, 2)
    assert mus[0, 0] == pytest.approx(mus[0, 1])

    with caplog.at_level(logging.WARNING):
        flory.prepare_reactions([1, -2], phi_means)
    assert "conservation" in caplog.text


def test_reactions_2comp_mu(caplog):
    """Test whether we can control internal energy differences via shifting the
    external reservoir potential.
    """
    num_comp = 2
    chis = np.full((num_comp, num_comp), 0.5)
    phi_means = [0.4, 0.6]
    w = 0.5

    ens, constr = flory.prepare_reactions([1, -1], phi_means, mus=[1, 1 + w])
    free_energy = flory.FloryHuggins(num_comp, chis)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction,
        free_energy.entropy,
        ensemble=ens,
        constraints=constr,
        random_std=1.0,  # use less aggressive randomness to avoid rapid dying of compartments
    )

    phases = finder.run(progress=False).get_clusters()
    assert phases.num_phases == 1
    assert sum(phases.fractions.flat) == pytest.approx(1)
    mus = free_energy.chemical_potentials(phases.fractions)
    assert mus.shape == (1, 2)
    assert mus[0, 0] == pytest.approx(mus[0, 1] - w)


def test_reactions_3comp(caplog):
    num_comp = 3
    chis = np.full((num_comp, num_comp), 0.5)
    phi_means = [0.4, 0.3, 0.3]
    sizes = [2.0, 1.0, 1.0]

    ens, constr = flory.prepare_reactions([0, 1, -1], phi_means, sizes=sizes)
    assert isinstance(ens, flory.SemiGrandCanonicalEnsemble)
    assert constr.num_cons == 1
    free_energy = flory.FloryHuggins(num_comp, chis, sizes)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction,
        free_energy.entropy,
        ensemble=ens,
        constraints=constr,
        random_std=1.0,  # use less aggressive randomness to avoid rapid dying of compartments
    )

    phases = finder.run(progress=False).get_clusters()
    assert phases.num_phases == 1
    assert sum(phases.fractions.flat) == pytest.approx(1)
    mus = free_energy.chemical_potentials(phases.fractions)
    assert phases.fractions[0, 0] == pytest.approx(0.4)
    assert phases.fractions[0, 1:].sum() == pytest.approx(0.6)
    assert mus.shape == (1, 3)
    assert mus[0, 1] == pytest.approx(mus[0, 2], abs=1e-4)

    with caplog.at_level(logging.WARNING):
        flory.prepare_reactions([0, 1, -2], phi_means)
    assert "conservation" in caplog.text


@pytest.mark.parametrize(
    "stoichiometries",
    [
        [[0, 1, -1, 0], [0, 0, 1, -1]],
        [[0, 1, -1, 0], [0, 0, 1, -1], [0, 1, 0, -1]],
    ],
)
def test_reactions_4comp(stoichiometries):
    num_comp = 4
    chis = np.full((num_comp, num_comp), 0.5)
    phi_means = [0.4, 0.2, 0.2, 0.2]
    sizes = [2.0, 1.0, 1.0, 1.0]

    ens, constr = flory.prepare_reactions(stoichiometries, phi_means, sizes=sizes)
    assert isinstance(ens, flory.SemiGrandCanonicalEnsemble)
    assert constr.num_cons == 1
    free_energy = flory.FloryHuggins(num_comp, chis, sizes)
    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction,
        free_energy.entropy,
        ensemble=ens,
        constraints=constr,
        random_std=1.0,  # use less aggressive randomness to avoid rapid dying of compartments
    )

    phases = finder.run(progress=False).get_clusters()

    assert phases.num_phases == 1
    assert sum(phases.fractions.flat) == pytest.approx(1)
    mus = free_energy.chemical_potentials(phases.fractions)
    assert phases.fractions[0, 0] == pytest.approx(0.4)
    assert phases.fractions[0, 1:].sum() == pytest.approx(0.6)
    assert mus.shape == (1, 4)
    assert mus[0, 1] == pytest.approx(mus[0, 2], abs=1e-4)
    assert mus[0, 1] == pytest.approx(mus[0, 3], abs=1e-4)
