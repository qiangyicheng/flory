"""
.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np
import pytest

import flory


@pytest.mark.parametrize("num_components", [4, 6, 8])
@pytest.mark.parametrize("chi", [8, 9, 10])
@pytest.mark.parametrize("size", [1.0, 1.2])
def test_find_coexisting_phases_symmetric(num_components: int, chi: float, size: float):
    """Test function `find_coexisting_phases` with a symmetric system"""
    chis = chi - np.identity(num_components) * chi
    phi_means = np.ones(num_components) / num_components
    sizes = np.ones(num_components) * size
    num_compartments = num_components * 4

    phi_l = np.exp(-size * chi)
    phi_h = 1.0 - (num_components - 1) * phi_l

    volumes_ref = np.ones(num_components) / num_components
    phis_ref = phi_l + np.identity(num_components) * (phi_h - phi_l)

    volumes_calc, phis_calc = flory.find_coexisting_phases(
        chis, phi_means, num_compartments, sizes=sizes, tolerance=1e-7, progress=True
    )
    np.testing.assert_allclose(volumes_calc, volumes_ref, rtol=1e-2, atol=1e-5)
    np.testing.assert_allclose(phis_calc, phis_ref, rtol=1e-2, atol=1e-5)


@pytest.mark.parametrize("num_compartments", [8, 16, 32])
def test_find_coexisting_phases_asymmetric_ternary(num_compartments: int):
    """Test function `find_coexisting_phases` with a symmetric system"""
    chis = np.array([[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]])
    phi_means = np.array([0.16, 0.55, 0.29])
    sizes = np.array([2.0, 2.0, 1.0])

    volumes_ref = np.array([0.63348903, 0.36651097])
    phis_ref = np.array(
        [[0.07578904, 0.81377563, 0.11043533], [0.30555285, 0.09408195, 0.60036519]]
    )

    volumes_calc, phis_calc = flory.find_coexisting_phases(
        chis, phi_means, num_compartments, sizes=sizes, tolerance=1e-7, progress=False
    )
    np.testing.assert_allclose(volumes_calc, volumes_ref, rtol=1e-5)
    np.testing.assert_allclose(phis_calc, phis_ref, rtol=1e-5)


def test_CoexistingPhasesFinder_ODT():
    """Test class `CoexistingPhasesFinder` with ODT of a binary system"""
    num_components = 2
    chi_start = 2.9999
    chi_end = 1.0

    chis = chi_start - np.identity(num_components) * chi_start
    phi_means = np.ones(num_components) / num_components
    num_compartments = num_components * 4

    finder = flory.CoexistingPhasesFinder(
        chis,
        phi_means,
        num_compartments,
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
        chis = chi - np.identity(num_components) * chi
        finder.chis = chis
        ans = finder.run()
        assert finder.diagnostics["max_abs_incomp"] < 1e-5
        assert finder.diagnostics["max_abs_omega_diff"] < 1e-5
        assert finder.diagnostics["max_abs_js_diff"] < 1e-5
        phi_h = ans[1][0, 0]
        phi_l = ans[1][0, 1]
        line_chi.append(chi)
        line_l.append(phi_l)
        line_h.append(phi_h)
        if np.abs(phi_l - phi_h) < 1e-3:
            chi_ODT = chi
            break

    np.testing.assert_allclose(chi_ODT, 1.9999)


def test_CoexistingPhasesFinder_set_parameters():
    """Test class `CoexistingPhasesFinder`.s functionality of resetting parameters"""
    num_components = 4
    chis = 3 - np.identity(num_components) * 3
    phi_means = np.ones(num_components) / num_components
    num_compartments = num_components * 4

    # not-square chi matrix
    with pytest.raises(ValueError):
        finder = flory.CoexistingPhasesFinder(chis[:-1], phi_means, num_compartments)

    # asymmetric chi matrix
    achis = chis.copy()
    achis[0, 1] += 1
    with pytest.raises(ValueError):
        finder = flory.CoexistingPhasesFinder(achis, phi_means, num_compartments)

    # incorrect phi_means shape
    with pytest.raises(ValueError):
        finder = flory.CoexistingPhasesFinder(chis, phi_means[:-1], num_compartments)

    # incorrect total phi_means is allowed by warned
    finder = flory.CoexistingPhasesFinder(chis, phi_means + 0.001, num_compartments)

    sizes = np.ones(num_components)
    # incorrect sizes shape
    with pytest.raises(ValueError):
        finder = flory.CoexistingPhasesFinder(
            chis, phi_means, num_compartments, sizes=sizes[:-1]
        )

    # incorrect sizes values are allowed by warned
    finder = flory.CoexistingPhasesFinder(
        chis, phi_means, num_compartments, sizes=sizes - 2
    )

    # use custom generator
    mtg = np.random.PCG64()
    finder = flory.CoexistingPhasesFinder(
        chis, phi_means, num_compartments, sizes=sizes, rng=np.random.Generator(mtg)
    )

    omegas = finder.omegas.copy()
    # reinitialize with additional noise
    newomegas = omegas + np.random.rand(omegas.shape[0], omegas.shape[1])
    finder.reinitialize_from_omegas(newomegas)
    np.testing.assert_allclose(newomegas, finder.omegas)

    # reinitialize with omegas with incorrect shape
    with pytest.raises(ValueError):
        finder.reinitialize_from_omegas(omegas[:-1])

    phis = finder.phis.copy()
    # reinitialize with additional noise
    newphis = phis + np.random.rand(phis.shape[0], phis.shape[1])
    finder.reinitialize_from_phis(newphis)

    # reinitialize with omegas with incorrect shape
    with pytest.raises(ValueError):
        finder.reinitialize_from_phis(phis[:-1])

    # reset chis with asymmetric chis
    with pytest.raises(ValueError):
        finder.chis = achis

    chis = finder.chis.copy()
    # reset chis with chis with incorrect shape
    with pytest.raises(ValueError):
        finder.chis = chis[:-1]

    # shift chis by a constant, which takes no effect to the calculation
    finder.chis = chis + 1
    np.testing.assert_allclose(chis + 1, finder.chis)

    phi_means = finder.phi_means.copy()
    # reset phi_means with incorrect shape
    with pytest.raises(ValueError):
        finder.phi_means = phi_means[:-1]

    # incorrect total phi_means is allowed by warned
    finder.phi_means = phi_means + 0.1
    np.testing.assert_allclose(phi_means + 0.1, finder.phi_means)

    sizes = finder.sizes.copy()
    # reset phi_means with incorrect shape
    with pytest.raises(ValueError):
        finder.sizes = sizes[:-1]

    # incorrect total phi_means is allowed by warned
    sizes[0] = 3
    finder.sizes = sizes
    np.testing.assert_allclose(sizes, finder.sizes)
