"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import stats

from flory.ensemble import CanonicalEnsemble


def test_canonical_ensemble():
    """test CanonicalEnsemble class"""
    with pytest.raises(ValueError):
        CanonicalEnsemble(0)

    e = CanonicalEnsemble(1, [1])
    assert e.num_comp == 1
    e = CanonicalEnsemble(2)
    assert e.num_comp == 2
    np.testing.assert_allclose(e.phi_means, [0.5, 0.5])
    e.phi_means = [1, 0]
    np.testing.assert_allclose(e.phi_means, [1, 0])
    with pytest.raises(ValueError):
        e.phi_means = [0.25, 0.25, 0.25, 0.25]
    with pytest.raises(ValueError):
        e.phi_means = [2, 1]
