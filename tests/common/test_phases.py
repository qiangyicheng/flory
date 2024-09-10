"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import stats

from flory.common.phases import get_uniform_random_composition


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
