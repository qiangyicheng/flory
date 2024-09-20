"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from flory.common.math import xlogx


def test_xlogx():
    """test xlogx function"""
    assert xlogx(0) == 0
    assert xlogx(0.5) == pytest.approx(0.5 * np.log(0.5))
    assert np.isnan(xlogx(-0.1))

    np.testing.assert_allclose(
        xlogx(np.array([-0.1, 0, 0.5])), [np.nan, 0, 0.5 * np.log(0.5)]
    )
