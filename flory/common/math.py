"""Module providing auxiliary mathematical functions.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
from numba import float64, vectorize


@vectorize([float64(float64)])
def xlogx(x):
    if x == 0:
        return 0
    elif x < 0:
        return np.nan
    else:
        return x * np.log(x)
