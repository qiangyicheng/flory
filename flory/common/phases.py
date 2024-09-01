"""Module providing the :class:`Phases` class, which captures information about phases.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import typing

import numpy as np
from scipy import cluster, spatial


class Phases(typing.NamedTuple):
    r"""Contains information about compositions and relative sizes of many phases.

    Attributes:
        Js (:class:`~numpy.ndarray`):
            1D array with shape :math:`N_\mathrm{p}`, containing the volume :math:`J_p`
            of each phase.
        phis (:class:`~numpy.ndarray`):
            2D array with shape :math:`N_\mathrm{p} \times N_\mathrm{c}`,
            containing the volume fractions of the components in each phase
            :math:`\phi_{p,i}`. The first dimension must be the same as
            :paramref:`Js`.
    """

    Js: np.ndarray
    phis: np.ndarray

    @property
    def num_phases(self) -> int:
        """int: number of phases."""
        return len(self.Js)

    @property
    def num_components(self) -> int:
        """int: number of components"""
        return self.phis.shape[1]

    @property
    def mean_phis(self) -> np.ndarray:
        """:class:`numpy.ndarray`: Mean fraction averaged over phases"""
        return self.Js @ self.phis  # assumes Js are normalized

    def check_consistency(self) -> None:
        """Checks whether the information is consistent"""
        if self.Js.ndim != 1:
            raise ValueError("Js must be a 1d array")
        if self.phis.ndim != 2:
            raise ValueError("phis must be a 2d array")
        if self.Js.shape[0] != self.phis.shape[0]:
            raise ValueError("Js and phis must have consistent first dimension")
        if not np.isclose(self.Js.sum(), 1):
            raise ValueError("Sum of Js must be 1")

    def sort(self) -> Phases:
        r"""Sort the phases according to the index of most concentrated components.

        Returns:
            :class:`Phases`: The sorted phases
        """
        enrich_indexes = np.argsort(self.phis)
        sorting_index = np.lexsort(np.transpose(enrich_indexes))
        return Phases(self.Js[sorting_index], self.phis[sorting_index])

    def get_clusters(self, dist: float = 1e-2) -> Phases:
        r"""Find clusters of compositions.

        Find unique phases from compartments by clustering. The returning results are sorted
        according to the index of most concentrated components.

        Args:
            dist (float):
                Cut-off distance for cluster analysis

        Returns:
            :class:`Phases`: The clustered and sorted phases
        """
        if self.num_phases < 2:
            return self  # nothing to do here

        # calculate distances between compositions
        dists = spatial.distance.pdist(self.phis)
        # obtain hierarchy structure
        links = cluster.hierarchy.linkage(dists, method="centroid")
        # flatten the hierarchy by clustering
        clusters = cluster.hierarchy.fcluster(links, dist, criterion="distance")
        cluster_phis = np.array(
            [self.phis[clusters == n, :].mean(axis=0) for n in np.unique(clusters)]
        )
        cluster_Js = np.array(
            [self.Js[clusters == n].sum(axis=0) for n in np.unique(clusters)]
        )
        cluster_Js /= cluster_Js.sum()

        # return sorted results
        return Phases(cluster_Js, cluster_phis).sort()
