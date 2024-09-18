"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from flory.common.phases import Phases, PhasesResult


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
