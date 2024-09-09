"""Module for Flory-Huggins block interaction energy of mixture.

The Flory-Huggins block interaction means that several components may have exactly
same interactions. Equivalently, the matrix :math:`\chi_{ij}` has blocked structure.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np

from ..common import *
from .base import InteractionBase
from .flory_huggins import FloryHugginsInteractionBase, FloryHugginsInteractionCompiled


class FloryHugginsBlockInteraction(FloryHugginsInteractionBase):
    r"""Class for Flory-Huggins block interaction energy of mixture.

    The particular form of interaction energy density reads

    .. math::
        f_\mathrm{interaction}(\{\phi_i\}) = \sum_{i,j=1}^{N_\mathrm{C}} \frac{\chi_{ij}}{2} \phi_i\phi_j

    where :math:`\phi_i` is the fraction of component :math:`i`, :math:`\chi_{ij}` is the
    Flory-Huggins interaction matrix. The blocked structure of :math:`\chi_{ij}` is
    defined as

    .. math::
        \chi_{ij} = \chi_{rs} \qquad : \qquad i \in B_r, j \in B_s \;,

    where :math:`B_r` and :math:`B_s` are the set of index of components that belong to
    feature :math:`r` and :math:`s`, respectively.

    """

    def __init__(
        self,
        num_feat: int,
        chis_feat: np.ndarray | float,
        num_comp_per_feat: np.ndarray | int = 1,
    ):
        r"""
        Args:
            num_feat:
                Number of features :math:`N_\mathrm{S}`.
            chis_feat:
                The Flory-Huggins interaction matrix of features :math:`\chi_{rs}`.
            num_comp_per_feat:
                The number of components in each feature. An integer indicates that this
                value is the same for all features.
        """
        self.num_feat = num_feat

        num_comp_per_feat = convert_and_broadcast(num_comp_per_feat, (num_feat,))
        self._num_comp_per_feat = num_comp_per_feat

        chis_feat = convert_and_broadcast(chis_feat, (num_feat, num_feat))
        # ensure that the chi matrix is symmetric
        if not np.allclose(chis_feat, chis_feat.T):
            self._logger.warning("Using symmetrized χ interaction-matrix")
        chis_feat = 0.5 * (chis_feat + chis_feat.T)
        self._chis_feat = chis_feat

        super().__init__(
            num_comp=num_comp_per_feat.sum(),
            chis=make_square_blocks(chis_feat, num_comp_per_feat),
        )
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def num_comp_per_feat(self) -> np.ndarray:
        r"""Number of components in each feature."""
        return self._num_comp_per_feat

    @property
    def chis(self) -> np.ndarray:
        r"""The Flory-Huggins interaction matrix of components :math:`\chi_{ij}`.

        This property should not be modified directly. Consider property :attr:`chis_feat`
        instead.
        """
        return self._chis

    @property
    def chis_feat(self) -> np.ndarray:
        r"""The Flory-Huggins interaction matrix of features :math:`\chi_{rs}`."""
        return self._chis_feat

    @chis_feat.setter
    def chis_feat(self, chis_feat_new: np.ndarray):
        chis_feat_new = convert_and_broadcast(
            chis_feat_new, (self._num_feat, self._num_feat)
        )
        # ensure that the chi matrix is symmetric
        if not np.allclose(chis_feat_new, chis_feat_new.T):
            self._logger.warning("Using symmetrized χ interaction-matrix")
        chis_feat_new = 0.5 * (chis_feat_new + chis_feat_new.T)
        self._chis_feat = chis_feat_new
        self._chis = make_square_blocks(chis_feat_new, self._num_comp_per_feat)

    def _compiled_impl(
        self, *, additional_chis_shift: float = 1.0
    ) -> FloryHugginsInteractionCompiled:
        """Implementation of creating a compiled interaction instance.

        This method overwrites the interface
        :meth:`~flory.entropy.base.InteractionBase._compiled_impl` in
        :class:`~flory.entropy.base.InteractionBase`.

        Args:
            additional_chis_shift:
                Shift of the entire chis matrix to improve the convergence by evolving
                towards incompressible system faster. This value should be larger than 0.
                This value only affects the numerics, not the actual physical system. Note
                that with very large value, the convergence will be slowed down, since the
                algorithm no longer have enough ability to temporarily relax the
                incompressibility.

        Returns:
            : Instance of :class:`FloryHugginsInteractionCompiled`.
        """

        return FloryHugginsInteractionCompiled(
            self._chis_feat, -self._chis_feat.min() + additional_chis_shift
        )
