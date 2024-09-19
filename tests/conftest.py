"""This file is used to configure the test environment when running py.test.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _setup_and_teardown():
    """Helper function adjusting environment before and after tests."""
    # ensure we use the Agg backend, so figures are not displayed
    plt.switch_backend("agg")
    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test
    yield

    # clean up open matplotlib figures after the test
    plt.close("all")


@pytest.fixture(autouse=False, name="rng")
def init_random_number_generators():
    """Get a random number generator and set the seed of the random number generator.

    The function returns an instance of :func:`~numpy.random.default_rng()` and
    initializes the default generators of both :mod:`numpy` and :mod:`numba`.
    """
    return np.random.default_rng(0)


def pytest_configure(config):
    """Add markers to the configuration."""
    config.addinivalue_line("markers", "slow: test runs slowly")


def pytest_addoption(parser):
    """Pytest hook to add command line options parsed by pytest."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked as `slow`",
    )


def pytest_collection_modifyitems(config, items):
    """Pytest hook to filter a collection of tests."""
    # parse options provided to py.test
    running_cov = config.getvalue("--cov")
    runslow = config.getoption("--runslow", default=False)

    # prepare markers
    skip_cov = pytest.mark.skip(reason="skipped during coverage run")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")

    # check each test item
    for item in items:
        if "no_cover" in item.keywords and running_cov:
            item.add_marker(skip_cov)
        if "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
