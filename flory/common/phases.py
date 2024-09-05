"""Module providing the :class:`Phases` class, which captures information about phases.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import typing

import numpy as np
from scipy import cluster, spatial


class Phases:
    """Contains information about compositions and relative sizes of many phases."""

    def __init__(self, volumes: np.ndarray, fractions: np.ndarray):
        r"""
        Args:
            volumes:
                1D array with shape :math:`N_\mathrm{P}`, containing the volume
                :math:`J_p` of each phase.
            fractions:
                2D array with shape :math:`N_\mathrm{P} \times N_\mathrm{C}`,
                containing the volume fractions of the components in each phase
                :math:`\phi_{p,i}`. The first dimension must be the same as
                :paramref:`volumes`.
        """
        volumes = np.asarray(volumes)
        fractions = np.asarray(fractions)
        if volumes.ndim != 1:
            raise ValueError("volumes must be a 1d array")
        if fractions.ndim != 2:
            raise ValueError("fractions must be a 2d array")
        if volumes.shape[0] != fractions.shape[0]:
            raise ValueError("volumes and fractions must have consistent first dimension")
        self.volumes = volumes
        self.fractions = fractions

    def __str__(self) -> str:
        return f"Phases(volumes={self.volumes}, fractions={self.fractions})"

    @property
    def num_phases(self) -> int:
        r"""Number of phases :math:`N_\mathrm{P}`."""
        return len(self.volumes)

    @property
    def num_components(self) -> int:
        r"""Number of components :math:`N_\mathrm{C}`."""
        return self.fractions.shape[1]

    @property
    def mean_fractions(self) -> np.ndarray:
        r"""Mean fraction averaged over phases :math:`\bar{\phi}_i`"""
        return self.volumes @ self.fractions / self.volumes.sum()

    def sort(self) -> Phases:
        """Sort the phases according to the index of most concentrated components.

        Returns:
            : The sorted phases.
        """
        enrich_indexes = np.argsort(self.fractions)
        sorting_index = np.lexsort(np.transpose(enrich_indexes))
        return Phases(self.volumes[sorting_index], self.fractions[sorting_index])

    def get_clusters(self, dist: float = 1e-2) -> Phases:
        r"""Find clusters of compositions.

        Find unique phases from compartments by clustering. The returning results are
        sorted according to the index of most concentrated components.

        Args:
            dist (float):
                Cut-off distance for cluster analysis.

        Returns:
            : The clustered and sorted phases.
        """
        if self.num_phases < 2:
            return self  # nothing to do here

        # calculate distances between compositions
        dists = spatial.distance.pdist(self.fractions)
        # obtain hierarchy structure
        links = cluster.hierarchy.linkage(dists, method="centroid")
        # flatten the hierarchy by clustering
        clusters = cluster.hierarchy.fcluster(links, dist, criterion="distance")
        cluster_fractions = np.array(
            [self.fractions[clusters == n, :].mean(axis=0) for n in np.unique(clusters)]
        )
        cluster_volumes = np.array(
            [self.volumes[clusters == n].sum(axis=0) for n in np.unique(clusters)]
        )
        cluster_volumes /= cluster_volumes.sum()

        # return sorted results
        return Phases(cluster_volumes, cluster_fractions).sort()
