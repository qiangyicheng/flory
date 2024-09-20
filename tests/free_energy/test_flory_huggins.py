"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

import flory
from flory.common.phases import get_uniform_random_composition


@pytest.mark.parametrize("num_comp", [1, 2, 3])
def test_flory_huggins_single_size(num_comp, rng):
    """Test some generic properties of the Flory-Huggins free energy"""
    chis = rng.normal(size=(num_comp, num_comp))
    chis += chis.T  # symmetrize
    fh = flory.FloryHuggins(num_comp, chis)

    phis = get_uniform_random_composition(num_comp, rng=rng)[np.newaxis, :]
    f = fh.free_energy_density(phis)
    assert f.shape == (1,)
    for solvent in range(num_comp):
        mu_bar = fh.exchange_chemical_potentials(phis, solvent)
        assert mu_bar.shape == (1, num_comp)
        pressure = np.sum(phis * mu_bar, axis=-1) - f
        np.testing.assert_allclose(fh.pressure(phis, solvent), pressure)


@pytest.mark.parametrize("num_comp", [1, 2, 3])
def test_flory_huggins_many_sizes(num_comp, rng):
    """Test some generic properties of the Flory-Huggins free energy"""
    chis = rng.normal(size=(num_comp, num_comp))
    chis += chis.T  # symmetrize
    sizes = rng.uniform(1, 2, size=(num_comp))
    fh = flory.FloryHuggins(num_comp, chis, sizes=sizes)

    phis = get_uniform_random_composition(num_comp, rng=rng)[np.newaxis, :]
    f = fh.free_energy_density(phis)
    assert f.shape == (1,)
    for solvent in range(num_comp):
        mu_bar = fh.exchange_chemical_potentials(phis, solvent)
        assert mu_bar.shape == (1, num_comp)
        pressure = np.sum(phis * mu_bar, axis=-1) - f
        np.testing.assert_allclose(fh.pressure(phis, solvent), pressure)
