"""
.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np
import pytest

import flory


@pytest.mark.parametrize("num_comp", [4, 5, 6])
@pytest.mark.parametrize("chi", [8, 9, 10])
@pytest.mark.parametrize("size", [1.0, 1.2])
def test_find_coexisting_phases_symmetric(num_comp: int, chi: float, size: float):
    """Test function `find_coexisting_phases` with a symmetric system"""
    chis = chi - np.identity(num_comp) * chi
    phi_means = np.ones(num_comp) / num_comp
    sizes = np.ones(num_comp) * size

    phi_l = np.exp(-size * chi)
    phi_h = 1.0 - (num_comp - 1) * phi_l

    volumes_ref = np.ones(num_comp) / num_comp
    phis_ref = phi_l + np.identity(num_comp) * (phi_h - phi_l)

    phases = flory.find_coexisting_phases(
        num_comp, chis, phi_means, sizes=sizes, tolerance=1e-7, progress=True
    )
    np.testing.assert_allclose(phases.volumes, volumes_ref, rtol=1e-2, atol=1e-5)
    np.testing.assert_allclose(phases.fractions, phis_ref, rtol=1e-2, atol=1e-5)


@pytest.mark.parametrize("num_part", [8, 16])
def test_find_coexisting_phases_asymmetric_ternary(num_part: int):
    """Test function `find_coexisting_phases` with a symmetric system"""
    num_comp = 3
    chis = np.array([[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]])
    phi_means = np.array([0.16, 0.55, 0.29])
    sizes = np.array([2.0, 2.0, 1.0])

    volumes_ref = np.array([0.63348903, 0.36651097])
    phis_ref = np.array(
        [[0.07578904, 0.81377563, 0.11043533], [0.30555285, 0.09408195, 0.60036519]]
    )

    phases = flory.find_coexisting_phases(
        num_comp,
        chis,
        phi_means,
        sizes=sizes,
        num_part=num_part,
        tolerance=1e-7,
        progress=False,
    )
    np.testing.assert_allclose(phases.volumes, volumes_ref, rtol=1e-5)
    np.testing.assert_allclose(phases.fractions, phis_ref, rtol=1e-5)


def test_CoexistingPhasesFinder_ODT():
    """Test class `CoexistingPhasesFinder` with ODT of a binary system"""
    num_comp = 2
    chi_start = 2.9999
    chi_end = 1.0

    chis = chi_start - np.identity(num_comp) * chi_start
    phi_means = np.ones(num_comp) / num_comp

    free_energy = flory.FloryHuggins(num_comp, chis)
    interaction = free_energy.interaction
    entropy = free_energy.entropy
    ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
    finder = flory.CoexistingPhasesFinder(
        interaction,
        entropy,
        ensemble,
        tolerance=1e-10,
        progress=False,
        additional_chis_shift=5,
        max_steps=1000000,
        interval=100000,
        acceptance_omega=0.02,
    )

    line_chi = []
    line_l = []
    line_h = []
    chi_ODT = 0.0
    for chi in np.arange(chi_start, chi_end, -0.1):
        chis = chi - np.identity(num_comp) * chi
        interaction.chis = np.array(chis)
        finder.set_interaction(interaction)
        phases = finder.run().get_clusters()
        assert finder.diagnostics["max_abs_incomp"] < 1e-5
        assert finder.diagnostics["max_abs_omega_diff"] < 1e-5
        assert finder.diagnostics["max_abs_js_diff"] < 1e-5
        phi_h = phases.fractions[0, 0]
        phi_l = phases.fractions[0, 1]
        line_chi.append(chi)
        line_l.append(phi_l)
        line_h.append(phi_h)
        if np.abs(phi_l - phi_h) < 1e-3:
            chi_ODT = chi
            break

    np.testing.assert_allclose(chi_ODT, 1.9999)
