"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np


def assert_phases_allclose(p1, p2, *, tol=1e-7) -> None:
    """Assert that two phase collections match up to phase ordering.

    The best phase permutation is selected by :meth:`Phases.match_phases`, then
    the distance for every matched phase is checked against an absolute tolerance
    of ``tol``.

    Args:
        p1: The first phase collection.
        p2: The phase collection to compare with ``p1``.
        tol: Base tolerance used for the matched phase distances.

    Raises:
        ValueError: If the phase collections cannot be matched, such as when their
            phase or component counts differ.
        AssertionError: If any matched phase distance exceeds the tolerance.
    """
    # determine best permutation to match phases
    permuted, dists = p1.match_phases(p2, ret_dists=True)

    # get total distance between phases and compare to tolerance
    phase_dists = dists[np.arange(p1.num_phases), permuted]
    np.testing.assert_allclose(phase_dists, 0, atol=tol)
