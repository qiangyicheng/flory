"""
.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np
import pytest

import flory


@pytest.mark.parametrize("num_comp", [3, 4, 5])
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

    fh = flory.FloryHuggins(num_comp, chis, sizes)
    exchange_mus = fh.exchange_chemical_potentials(phases.fractions, 0)
    for mus in exchange_mus:
        np.testing.assert_allclose(mus, exchange_mus[0], rtol=1e-2, atol=1e-5)
    pressures = fh.pressure(phases.fractions, 0)
    for p in pressures:
        np.testing.assert_allclose(p, pressures[0], rtol=1e-2, atol=1e-5)
    modes = fh.num_unstable_modes(phases.fractions)
    np.testing.assert_equal(modes.sum(), 0)


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
