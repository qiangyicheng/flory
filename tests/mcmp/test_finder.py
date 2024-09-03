"""
.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np
import pytest

import flory


def test_CoexistingPhasesFinder_set_instances():
    num_comp = 3
    chis = np.array([[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]])
    phi_means = np.array([0.16, 0.68, 0.16])
    sizes = np.array([2.0, 2.0, 1.0])

    free_energy = flory.FloryHuggins(num_comp, chis, sizes)
    ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
    constraint_1 = flory.LinearLocalConstraint(num_comp, [-1.0, 0.0, 1.0], 0.0)
    constraint_2 = flory.LinearGlobalConstraint(num_comp, [-1.0, 0.0, 1.0], 0.0)

    finder = flory.CoexistingPhasesFinder(
        free_energy.interaction, free_energy.entropy, ensemble, [constraint_1]
    )

    finder.set_constraints([constraint_1, constraint_2])

    finder.run(max_steps=10000)

    new_chis = free_energy.chis.copy()
    new_chis[0, 1] += 0.5
    free_energy.chis = new_chis

    new_sizes = free_energy.sizes.copy()
    new_sizes[0] += 0.5
    free_energy.sizes = new_sizes

    new_ensemble = flory.GrandCanonicalEnsemble.from_chemical_potential(
        num_comp, [0.1] * num_comp, new_sizes
    )

    finder.set_interaction(free_energy.interaction)
    finder.set_entropy(free_energy.entropy)
    finder.set_ensemble(new_ensemble)
    finder.set_constraints([flory.NoConstraint(num_comp)])

    finder.run(max_steps=10000)
