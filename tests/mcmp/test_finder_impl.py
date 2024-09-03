"""
.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np
import pytest
from numba.experimental import jitclass

from flory.mcmp._finder_impl import *


def test_count_valid_phases():
    """Test function count_valid_phases()"""
    Js = np.array([1.0, 0.3, 0.2, 0.8, -0.2])
    assert count_valid_compartments(Js, 1.0) == 0
    assert count_valid_compartments(Js, 0.1) == 4
    assert count_valid_compartments(Js, 0.2) == 3
    assert count_valid_compartments(Js, -0.4) == 5


def test_make_valid_phase_masks():
    """Test function make_valid_phase_masks()"""
    Js = np.array([1.0, 0.3, 0.2, 0.8, -0.2])
    np.testing.assert_equal(
        make_valid_compartment_masks(Js, 1.0),
        np.array([False, False, False, False, False]),
    )
    np.testing.assert_equal(
        make_valid_compartment_masks(Js, 0.1), np.array([True, True, True, True, False])
    )
    np.testing.assert_equal(
        make_valid_compartment_masks(Js, 0.2),
        np.array([True, True, False, True, False]),
    )
    np.testing.assert_equal(
        make_valid_compartment_masks(Js, -0.4), np.array([True, True, True, True, True])
    )


@pytest.mark.parametrize("threshold", [0, 0.1, 0.2])
@pytest.mark.parametrize("scaler", [1.0, 0.8, 10.0])
def test_revive_compartments_by_random(threshold: float, scaler: float):
    """Test function revive_compartments_by_random()"""
    amp = 3.0
    rng = np.random.default_rng()
    Js = np.array([1.0, 0.3, 0.2, 1.8, -0.2, 1.7, 2.2])
    targets = rng.uniform(-amp, amp, (3, 7))
    target_centers = 0.5 * (targets.min(axis=1) + targets.max(axis=1))
    mask = Js > threshold
    Js_original = Js.copy()
    targets_original = targets.copy()
    oldlist = {str(targets_original[:, itr]) for itr, flag in enumerate(mask) if flag}
    revive_compartments_by_random(Js, targets, threshold, rng, scaler)
    for itr, flag in enumerate(mask):
        if flag:
            assert Js_original[itr] == Js[itr]
            assert np.all(targets_original[:, itr] == targets[:, itr])
        else:
            assert np.abs(targets[:, itr] - target_centers).max() < scaler * 3.0
            assert Js[itr] == 1
            assert str(targets[:, itr]) not in oldlist


@pytest.mark.parametrize("threshold", [0, 0.1, 0.2, 0.3])
def test_revive_compartments_by_copy(threshold: float):
    """Test function revive_compartments_by_copy()"""
    amp = 3.0
    rng = np.random.default_rng()
    Js = np.array([1.0, 0.3, 0.2, 1.8, -0.2, 1.7, 2.2])
    targets = rng.uniform(-amp, amp, (3, 7))
    mask = Js > threshold
    Js_original = Js.copy()
    targets_original = targets.copy()
    oldlist = {str(targets_original[:, itr]) for itr, flag in enumerate(mask) if flag}
    revive_compartments_by_copy(Js, targets, threshold, rng)
    for itr, flag in enumerate(mask):
        if flag:
            assert Js_original[itr] >= Js[itr]
            assert np.all(targets_original[:, itr] == targets[:, itr])
        else:
            assert str(targets[:, itr]) in oldlist
    np.testing.assert_allclose(np.sum(Js), np.sum(Js_original * mask))
    np.testing.assert_allclose(
        np.dot(targets, Js), np.dot(targets_original, Js_original * mask)
    )


def test_revive_compartments_by_copy_not_nice():
    """Test function revive_compartments_by_copy()"""
    amp = 3.0
    threshold = 0.2

    @jitclass()
    class FakeRNG:
        def __init__(self):
            return

        @staticmethod
        def integers(arg1, arg2):
            return 0

    rng = np.random.default_rng()
    fakerng = FakeRNG()
    Js = np.array([0.0, 0.0, 0.5, 3.5])
    targets = rng.uniform(-amp, amp, (3, len(Js)))
    mask = Js > threshold
    Js_original = Js.copy()
    targets_original = targets.copy()
    oldlist = {str(targets_original[:, itr]) for itr, flag in enumerate(mask) if flag}
    revive_compartments_by_copy(Js, targets, threshold, fakerng)
    for itr, flag in enumerate(mask):
        if flag:
            assert Js_original[itr] >= Js[itr]
            assert np.all(targets_original[:, itr] == targets[:, itr])
        else:
            assert str(targets[:, itr]) in oldlist
    np.testing.assert_allclose(np.sum(Js), np.sum(Js_original * mask))
    np.testing.assert_allclose(
        np.dot(targets, Js), np.dot(targets_original, Js_original * mask)
    )
