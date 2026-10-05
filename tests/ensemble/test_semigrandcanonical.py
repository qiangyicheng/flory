"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import stats

from flory.ensemble import SemiGrandCanonicalEnsemble


def test_semigrandcanonical_ensemble():
    """test SemiGrandCanonicalEnsemble class"""
    with pytest.raises(ValueError):
        SemiGrandCanonicalEnsemble(0, True, 1)

    e = SemiGrandCanonicalEnsemble(2, True, [1, 1])
    assert e.num_comp == 2
    np.testing.assert_array_equal(e.is_canonical, np.ones(2, dtype=bool))

    e = SemiGrandCanonicalEnsemble(2, [True, False], [1, 1])
    assert e.num_comp == 2
    np.testing.assert_array_equal(e.is_canonical, np.array([True, False], dtype=bool))
    e.is_canonical = True
    np.testing.assert_array_equal(e.is_canonical, np.ones(2, dtype=bool))

    with pytest.raises(ValueError):
        e.constraint = [0.6, 0.7, 0.3]
