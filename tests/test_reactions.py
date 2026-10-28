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
    assert mus[0, 0] == pytest.approx(mus[0, 1], abs=1e-4)

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
    assert sum(phases.fractions.flat) == pytest.approx(1, abs=1e-4)
    mus = free_energy.chemical_potentials(phases.fractions)
    assert mus.shape == (1, 2)
    assert mus[0, 0] == pytest.approx(mus[0, 1] - w, abs=1e-4)


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
    assert sum(phases.fractions.flat) == pytest.approx(1, abs=1e-4)
    mus = free_energy.chemical_potentials(phases.fractions)
    assert phases.fractions[0, 0] == pytest.approx(0.4, abs=1e-4)
    assert phases.fractions[0, 1:].sum() == pytest.approx(0.6, abs=1e-4)
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
    assert sum(phases.fractions.flat) == pytest.approx(1, abs=1e-4)
    mus = free_energy.chemical_potentials(phases.fractions)
    assert phases.fractions[0, 0] == pytest.approx(0.4, abs=1e-4)
    assert phases.fractions[0, 1:].sum() == pytest.approx(0.6, abs=1e-4)
    assert mus.shape == (1, 4)
    assert mus[0, 1] == pytest.approx(mus[0, 2], abs=1e-4)
    assert mus[0, 1] == pytest.approx(mus[0, 3], abs=1e-4)


def test_reaction_stoichiometry_uses_volume_fractions(caplog):
    phi_means = [0.2, 0.4, 0.4]
    sizes = [1.0, 2.0, 1.0]
    stoichiometric_matrix = [[-1, 1, 0]]

    with caplog.at_level(logging.WARNING):
        _, constraint = flory.prepare_reactions(
            stoichiometric_matrix, phi_means, sizes=sizes
        )

    assert "volume conservation" not in caplog.text
    assert constraint.num_cons == 1
    np.testing.assert_allclose(
        constraint.Cs @ np.asarray(stoichiometric_matrix).T, 0, atol=1e-10
    )
    np.testing.assert_allclose(constraint.Ts, constraint.Cs @ phi_means)


def test_reaction_equilibrium():
    num_comp = 2
    chis = np.zeros((num_comp, num_comp))
    phi_means = [0.4, 0.6]
    sigma = [1, -1]
    ws = [0, 2]

    phases = flory.find_coexisting_phases(
        num_comp, chis, phi_means, stoichiometric_matrix=sigma, internal_energies=ws
    )

    assert phases.num_phases == 1
    phi0, phi1 = phases.fractions[0]
    assert phi0 / phi1 == pytest.approx(np.exp(2.0), rel=1e-4)


def test_reaction_equilibrium_phasesep():
    num_comp = 3
    chis = np.zeros((num_comp, num_comp))
    chis[0, 1:] = chis[1:, 0] = 3
    phi_means = [0.4, 0.4, 0.2]
    sigma = [0, 1, -1]
    ws = [0, 0, 1.5]

    phases = flory.find_coexisting_phases(
        num_comp, chis, phi_means, stoichiometric_matrix=sigma, internal_energies=ws
    )

    assert phases.num_phases == 2
    assert phases.mean_fractions[0] == pytest.approx(0.4, abs=1e-4)
    np.testing.assert_allclose(
        phases.fractions[:, 1], np.exp(1.5) * phases.fractions[:, 2]
    )


def test_reaction_equilibrium_3reacts():
    num_comp = 3
    chis = np.zeros((num_comp, num_comp))
    phi_means = [0.2, 0.2, 0.6]
    sigma = [[0, 1, -1], [1, 0, -1], [1, -1, 0]]
    sizes = [1, 1, 1]
    ws = [0, 1, 2]

    phases = flory.find_coexisting_phases(
        num_comp,
        chis,
        phi_means,
        stoichiometric_matrix=sigma,
        sizes=sizes,
        internal_energies=ws,
    )

    assert phases.num_phases == 1
    fs = np.exp(-np.array(ws))
    np.testing.assert_allclose(phases.fractions[0], fs / fs.sum())


def test_reactions_interactions():
    """test reactions with interactions"""
    num_comp = 2
    chis = np.array([[0, 4], [4, 0]])
    phi_means = [0.4, 0.6]
    sigma = [1, -1]
    ws = [0, 0.1]

    phases = flory.find_coexisting_phases(
        num_comp, chis, phi_means, stoichiometric_matrix=sigma, internal_energies=ws
    )

    assert phases.num_phases == 1
    f = flory.FloryHuggins(2, chis=chis)
    mu1, mu2 = f.chemical_potentials(phases.fractions)[0]
    assert mu1 == pytest.approx(mu2 + 0.1, abs=1e-4)
