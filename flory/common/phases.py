"""Module providing common utilities to deal with phases and their composition.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import typing

import numpy as np
from scipy import cluster, spatial


def get_uniform_random_composition(num_comps: int, rng=None) -> np.ndarray:
    """pick concentrations uniformly from allowed simplex (sum of fractions < 1)

    Args:
        num_comps (int): the number of components to use
        rng: The random number generator

    Returns:
        An array with `num_comps` random fractions
    """
    rng = np.random.default_rng(rng)

    phis = np.empty(num_comps)
    phi_max = 1.0
    for d in range(num_comps - 1):
        x = rng.beta(1, num_comps - d - 1) * phi_max
        phi_max -= x
        phis[d] = x
    phis[-1] = 1 - phis[:-1].sum()
    return phis


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

    def _copy(self, volumes: np.ndarray, fractions: np.ndarray) -> Phases:
        """create copy with changed volume and fraction

        This method helps with subclassing methods that should keep other information
        intact.
        """
        return self.__class__(volumes, fractions)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(volumes={self.volumes}, fractions={self.fractions})"

    @property
    def num_phases(self) -> int:
        r"""Number of phases :math:`N_\mathrm{P}`."""
        return len(self.volumes)

    @property
    def num_components(self) -> int:
        r"""Number of components :math:`N_\mathrm{C}`."""
        return self.fractions.shape[1]

    @property
    def total_volume(self) -> float:
        """Total volume of entire system"""
        return self.volumes.sum()

    @property
    def mean_fractions(self) -> np.ndarray:
        r"""Mean fraction averaged over phases :math:`\bar{\phi}_i`"""
        return self.volumes @ self.fractions / self.total_volume

    def normalize(self) -> Phases:
        """normalize the phases so that their volumes adds to one"""
        return self._copy(self.volumes / self.total_volume, self.fractions)

    def sort(self) -> Phases:
        """Sort the phases according to the index of most concentrated components.

        Returns:
            : The sorted phases.
        """
        enrich_indexes = np.argsort(self.fractions)
        sorting_index = np.lexsort(np.transpose(enrich_indexes))
        return self._copy(self.volumes[sorting_index], self.fractions[sorting_index])

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

        # collect data for each cluster
        cluster_fractions = []
        cluster_volumes = []
        for n in np.unique(clusters):
            current_fractions = self.fractions[clusters == n, :]
            current_volumes = self.volumes[clusters == n]
            mean_fractions = np.average(
                current_fractions, weights=current_volumes, axis=0
            )
            cluster_fractions.append(mean_fractions)
            cluster_volumes.append(current_volumes.sum())

        return self._copy(cluster_volumes, cluster_fractions)


class PhasesResult(Phases):
    """Contains compositions and relative sizes of many phases along with extra information."""

    def __init__(
        self, volumes: np.ndarray, fractions: np.ndarray, *, info: dict | None = None
    ):
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
            info:
                Additional information about how the phases were obtained.
        """
        super().__init__(volumes, fractions)
        self._info = {} if info is None else info

    @classmethod
    def from_phases(cls, phases: Phases, *, info: dict | None = None) -> PhasesResult:
        r"""create phase result from :class:`Phases`

        Args:
            phases:
                The :class:`Phases` containing volumes and fractions.
            info:
                Additional information about how the phases were obtained
        """
        return cls(phases.volumes, phases.fractions, info=info)

    @property
    def info(self) -> dict:
        r"""Information for the current collection of phases."""
        return self._info

    def _copy(self, volumes: np.ndarray, fractions: np.ndarray) -> PhasesResult:
        """create copy with changed volume and fraction

        This method helps with subclassing methods that should keep other information
        intact.
        """
        return self.__class__(volumes, fractions, info=self.info.copy())
