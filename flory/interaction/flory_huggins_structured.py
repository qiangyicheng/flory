r"""Module for Flory-Huggins energy with structured interactions.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np

from ..common import *
from .base import InteractionBase
from .flory_huggins import FloryHugginsInteractionBase, FloryHugginsInteractionCompiled


def _get_chi_matrix_from_features(
    features: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """calculate Flory matrix from feature matrix and associated weights

    Args:
        features (np.ndarray):
            Feature matrix of all components
        weights (np.ndarray):
            Weights of the individual features

    Returns:
        np.ndarray: the corresponding chi matrix
    """
    eijs = np.einsum("in,jn,n->ij", features, features, weights)
    eij_diag = np.diag(eijs)
    chis = eijs - 0.5 * (eij_diag[:, np.newaxis] + eij_diag[np.newaxis, :])
    return chis


class FloryHugginsStructured(FloryHugginsInteractionBase):
    r"""Class for Flory-Huggins structured interactions.

    The particular form of interaction energy density reads

    .. math::
        f_\mathrm{interaction}(\{\phi_i\}) = \sum_{i,j=1}^{N_\mathrm{C}} \frac{\chi_{ij}}{2} \phi_i\phi_j

    where :math:`\phi_i` is the fraction of component :math:`i`, :math:`\chi_{ij}` is the
    Flory-Huggins interaction matrix. The main idea of the structured interaction matrix
    is that each of the :math:`N_\mathrm{C}` components has :math:`N_\mathrm{F}`
    features summarized by the feature matrix :math:`s_{i, \alpha}`:

    .. math::
        \chi_{ij} = \frac12 \sum_{\alpha=1}^{N_\mathrm{F}} w_\alpha
                                \left(s_{i,\alpha} - s_{j, \alpha}\right)^2

    implying the rank of :math:`\chi_{ij}` is at most :math:`N_\mathrm{C}`. The weight
    vector :math:`w_\alpha` allows implementing attractive and repulsive features.
    """

    def __init__(self, features: np.ndarray, weights: np.ndarray | float = 1):
        r"""
        Args:
            features (np.ndarray):
                Feature matrix :math:`s_{i, \alpha}` of all components
            weights (np.ndarray):
                Weights :math:`w_\alpha` of the individual features
        """
        self._features = np.atleast_2d(features)
        self._weights = np.broadcast_to(weights, self._features.shape[1])
        chis = _get_chi_matrix_from_features(self._features, self._weights)
        super().__init__(num_comp=self._features.shape[0], chis=chis)
        self._logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def from_random_normal(
        cls,
        num_comp: int,
        num_feat: int,
        *,
        feature_mean: float = 0,
        feature_std: float = 1,
        weight_mean: float = 1,
        weight_std: float = 0,
        rng: np.random.Generator | None = None,
    ) -> FloryHugginsStructured:
        """Create normal random structured interaction matrix

        Args:
            num_comp (int):
                Component count
            num_feat (int):
                Feature count
            feature_mean (float):
                Mean value of the normal distribution of the features
            feature_std (float):
                Standard deviation of the normal distribution of the features
            weight_mean (float):
                Mean value of the normal distribution of the feature weights
            weight_std (float):
                Standard deviation of the normal distribution of the feature weights
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
        """
        rng = np.random.default_rng(rng)
        features = rng.normal(
            loc=feature_mean, scale=feature_std, size=(num_comp, num_feat)
        )
        weights = rng.normal(loc=weight_mean, scale=weight_std, size=num_feat)
        return cls(features, weights=weights)

    @classmethod
    def from_random_uniform(
        cls,
        num_comp: int,
        num_feat: int,
        *,
        feature_range: tuple[float, float] = (-1, 1),
        weight_range: tuple[float, float] = (1, 1),
        rng: np.random.Generator | None = None,
    ) -> FloryHugginsStructured:
        """Create uniformly random structured interaction matrix

        Args:
            num_comp (int):
                Component count
            num_feat (int):
                Feature count
            feature_range (tuple of two floats):
                Minimal and maximal value of the uniform distribution of the features
            weight_range (tuple of two floats):
                Minimal and maximal value of the uniform distribution of the weights
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
        """
        rng = np.random.default_rng(rng)
        features = rng.uniform(*feature_range, size=(num_comp, num_feat))
        weights = rng.uniform(*weight_range, size=num_feat)
        return cls(features, weights=weights)

    @property
    def num_feat(self) -> int:
        r"""Number of feature for each component"""
        return self._features.shape[1]

    @property
    def features(self) -> np.ndarray:
        r"""Feature matrix"""
        return self._features

    @features.setter
    def features(self, features: np.ndarray):
        self._features = np.broadcast_to(features, (self.num_comp, self.num_feat))
        self._chis = _get_chi_matrix_from_features(self._features, self._weights)

    @property
    def weights(self) -> np.ndarray:
        r"""Feature weights"""
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray):
        self._weights = np.broadcast_to(weights, (self.num_feat,))
        self._chis = _get_chi_matrix_from_features(self._features, self._weights)
