"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

import flory
from flory.interaction.flory_huggins_structured import _get_chi_matrix_from_features


def test_get_chi_matrix_from_features(rng):
    """Test generation of the chi matrix"""
    num_comp = 10
    num_feat = 3

    features = rng.normal(size=(num_comp, num_feat))
    weights = rng.normal(size=(num_feat,))

    eijs = np.einsum("in,jn,n->ij", features, features, weights)
    eij_diag = np.diag(eijs)
    chis = eijs - 0.5 * (eij_diag[:, np.newaxis] + eij_diag[np.newaxis, :])
    np.testing.assert_allclose(chis, _get_chi_matrix_from_features(features, weights))
