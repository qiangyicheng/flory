"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib as mpl
import numpy as np
import pytest

from flory.common.phases import Phases, PhasesResult


def test_plotting():
    """A simple test to check whether the plotting method of PhasesResult exits normally and returns a QuadMesh object. 
    This test does not check the actual content of the plot, but only that it can be generated without errors."""
    #create mock data
    Nc, Np = 3, 2
    fractions = np.random.uniform(0, 1, size=(Np, Nc))
    fractions /= fractions.sum(axis=0)  # normalize
    volumes = [0.4, 0.8]
    phases = Phases(volumes, fractions)
    phases_res = PhasesResult.from_phases(phases)

    #invoke the plotting method and check that it returns a QuadMesh object
    image = phases_res.plot()
    assert isinstance(image, mpl.collections.QuadMesh)

    #Now check that it also works when passing an axes object
    fig, ax = mpl.pyplot.subplots()
    image = phases_res.plot(ax=ax)
    assert isinstance(image, mpl.collections.QuadMesh)

    #Now check that we can also pass additional keyword arguments to the plotting method
    image = phases_res.plot(ax=ax, cmap="viridis", vmin=0.1, vmax=0.3)
    assert isinstance(image, mpl.collections.QuadMesh)
