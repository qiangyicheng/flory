"""
.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np
import pytest

import mcmp

@pytest.mark.parametrize("num_components", [4, 6, 8])
@pytest.mark.parametrize("chi", [8, 9, 10])
@pytest.mark.parametrize("size", [1.0, 1.2])
def test_cpfinder_symmetric(num_components: int, chi:float, size:float):
    """Test function cpfinder() with a symmetric system"""
    chis = chi - np.identity(num_components) * chi
    phi_means = np.ones(num_components) / num_components
    sizes = np.ones(num_components) * size
    num_compartments = num_components * 4
    
    phi_l = np.exp(-size * chi)
    phi_h = 1. - (num_components-1)*phi_l

    volumes_ref = np.ones(num_components) / num_components
    phis_ref = phi_l + np.identity(num_components) * (phi_h - phi_l)

    volumes_calc, phis_calc = mcmp.cpfinder(
        chis, phi_means, num_compartments, sizes = sizes, tolerance=1e-7, progress=False
    )
    np.testing.assert_allclose(volumes_calc, volumes_ref, rtol=1e-2, atol=1e-5)
    np.testing.assert_allclose(phis_calc, phis_ref, rtol=1e-2, atol=1e-5)


@pytest.mark.parametrize("num_compartments", [8, 16, 32])
def test_cpfinder_asymmetric_ternary(num_compartments: int):
    """Test function cpfinder() with a symmetric system"""
    chis = np.array([[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]])
    phi_means = np.array([0.16, 0.55, 0.29])
    sizes = np.array([2.0, 2.0, 1.0])

    volumes_ref = np.array([0.63348903, 0.36651097])
    phis_ref = np.array(
        [[0.07578904, 0.81377563, 0.11043533], [0.30555285, 0.09408195, 0.60036519]]
    )

    volumes_calc, phis_calc = mcmp.cpfinder(
        chis, phi_means, num_compartments, sizes=sizes, tolerance=1e-7, progress=False
    )
    np.testing.assert_allclose(volumes_calc, volumes_ref, rtol=1e-2, atol=1e-5)
    np.testing.assert_allclose(phis_calc, phis_ref, rtol=1e-2, atol=1e-5)
