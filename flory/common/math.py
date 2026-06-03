"""Module providing auxiliary mathematical functions.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
from numba import float64, vectorize


@vectorize([float64(float64)])
def xlogx(x):
    r"""Calculate :math:`x\ln(x)` with a finite value at :math:`x=0`.

    Args:
        x:
            Input value.

    Returns:
        :math:`x\ln(x)` for positive :paramref:`x`, 0 for :paramref:`x` equal to
        zero, and :data:`numpy.nan` for negative :paramref:`x`.
    """
    if x == 0:
        return 0
    elif x < 0:
        return np.nan
    else:
        return x * np.log(x)
