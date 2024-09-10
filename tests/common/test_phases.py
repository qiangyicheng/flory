"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import stats

from flory.common.phases import Phases, PhasesResult, get_uniform_random_composition


@pytest.mark.parametrize("cls", [Phases, PhasesResult])
def test_phases(cls):
    """test basic properties of the Phases class"""
    Nc, Np = 3, 2
    fractions = np.random.uniform(0, 1, size=(Np, Nc))
    fractions /= fractions.sum(axis=0)  # normalize
    volumes = [0.4, 0.8]
    phases = Phases(volumes, fractions)
    if cls == PhasesResult:
        phases = PhasesResult.from_phases(phases, info={"test": "value"})

    assert phases.num_phases == Np
    assert phases.num_components == Nc
    assert phases.total_volume == pytest.approx(1.2)

    assert phases.fractions is fractions  # assume no copy!
    assert phases.volumes is not volumes  # needs to be copied
    np.testing.assert_allclose(phases.volumes, volumes)
    np.testing.assert_allclose(
        phases.mean_fractions, (0.4 * fractions[0] + 0.8 * fractions[1]) / 1.2
    )

    p2 = phases.get_clusters()
    assert isinstance(p2, cls)
    assert p2 is not phases
    if p2.volumes[1] > p2.volumes[0]:
        np.testing.assert_allclose(p2.volumes, phases.volumes)
        np.testing.assert_allclose(p2.fractions, phases.fractions)
    else:
        np.testing.assert_allclose(p2.volumes[::-1], phases.volumes)
        np.testing.assert_allclose(p2.fractions[::-1], phases.fractions)
    if cls == PhasesResult:
        assert p2.info["test"] == "value"

    p2 = phases.sort()
    assert isinstance(p2, cls)
    assert p2 is not phases
    if p2.volumes[1] > p2.volumes[0]:
        np.testing.assert_allclose(p2.volumes, phases.volumes)
        np.testing.assert_allclose(p2.fractions, phases.fractions)
    else:
        np.testing.assert_allclose(p2.volumes[::-1], phases.volumes)
        np.testing.assert_allclose(p2.fractions[::-1], phases.fractions)
    if cls == PhasesResult:
        assert p2.info["test"] == "value"

    p2 = phases.normalize()
    assert isinstance(p2, cls)
    assert p2 is not phases
    assert p2.total_volume == pytest.approx(1.0)
    if p2.volumes[1] > p2.volumes[0]:
        np.testing.assert_allclose(p2.volumes, phases.volumes / 1.2)
        np.testing.assert_allclose(p2.fractions, phases.fractions)
    else:
        np.testing.assert_allclose(p2.volumes[::-1], phases.volumes / 1.2)
        np.testing.assert_allclose(p2.fractions[::-1], phases.fractions)
    if cls == PhasesResult:
        assert p2.info["test"] == "value"


@pytest.mark.parametrize("cls", [Phases, PhasesResult])
def test_phases_wrong_input(cls):
    """test basic properties of the Phases class"""
    with pytest.raises(ValueError):
        cls(1, [])
    with pytest.raises(ValueError):
        cls([[1]], [])
    with pytest.raises(ValueError):
        cls([1], [1])
    with pytest.raises(ValueError):
        cls([1], [[[1]]])
    with pytest.raises(ValueError):
        cls([1], [[1], [2]])


@pytest.mark.parametrize("num_comps", [1, 2, 3])
def test_get_uniform_random_composition(num_comps):
    """test get_uniform_random_composition function"""
    phis = get_uniform_random_composition(num_comps)
    assert phis.shape == (num_comps,)
    assert phis.sum() == pytest.approx(1)


def test_get_uniform_random_composition_dist():
    """test distribution of get_uniform_random_composition"""
    phis = np.array([get_uniform_random_composition(3) for _ in range(1000)])
    assert stats.ks_2samp(phis[:, 0], phis[:, 1]).pvalue > 0.05
