"""Module for semi-grand canonical ensemble of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np
from numba import bool_, float64, int32
from numba.experimental import jitclass
from numpy.typing import NDArray

from .base import EnsembleBase, EnsembleBaseCompiled


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_is_canonical", bool_[::1]),  # a boolean array
        ("_constraint", float64[::1]),  # a C-continuous array
    ]
)
class SemiGrandCanonicalEnsembleCompiled(EnsembleBaseCompiled):
    r"""Compiled class for semi-grand canonical ensemble.

    In the semi-grand canonical ensemble, the either the average volume or the original
    chemical potentials of components are fixed. Since (translational) entropy is always
    defined for each component, this class is only aware of the component-based
    description of the system.

    If the average volume is fixed, the volume fractions distribution of the components
    in compartments can be obtained by normalizing the Boltzmann factors according to
    the average volume fractions,

        .. math::
            \phi_i^{(m)} &= \frac{\bar{\phi}_i}{Q_i} p_i^{(m)} \\
            Q_i &= \sum_m p_i^{(m)} J_m .

    In the case where the chemical potential is controlled, the volume fractions
    distribution of the components in compartments can be obtained by scaling the
    Boltzmann factors according to the scaled activity,

        .. math::
            \phi_i^{(m)} &= l_i e^{l_i \mu_i} p_i^{(m)} \\

    where :math:`l_i e^{l_i \mu_i}` is the scaled activity, :math:`l_i` is the relative
    volumes of molecules and :math:`\mu_i` is the chemical potentials of the components
    by volume.
    """

    def __init__(self, is_canonical: NDArray[bool], constraint: np.ndarray):
        r"""
        Args:
            is_canonical:
                1D array with the size of :math:`N_\mathrm{C}`, determining which
                components are treated canonically. The number of components
                :math:`N_\mathrm{C}` is inferred from this array.
            constraint:
                1D array with the size of :math:`N_\mathrm{C}`, containing the scaled
                activities of the components, :math:`l_i e^{l_i \mu_i}`.
        """
        self._num_comp = is_canonical.shape[0]
        self._is_canonical = is_canonical
        self._constraint = constraint  # do not affect chis

    @property
    def num_comp(self):
        return self._num_comp

    def normalize(
        self, phis_comp: np.ndarray, Qs: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        r"""Scale Boltzmann factors to component fractions.

        Args:
            phis_comp:
                Mutable component Boltzmann factors, updated in-place to component
                fractions.
            Qs:
                Single molecule partition functions of components. This parameter is
                unused for grand canonical normalization.
            masks:
                Masks indicating whether compartments are active.

        Returns:
            The incompressibility in each compartment.
        """
        incomp = -1.0 * np.ones_like(phis_comp[0])
        for itr_comp in range(self._num_comp):
            if self._is_canonical[itr_comp]:
                # treat component as canonical
                factor = self._constraint[itr_comp] / Qs[itr_comp]
                phis_comp[itr_comp] = factor * phis_comp[itr_comp] * masks
            else:
                # treat component as grand-canonical
                phis_comp[itr_comp] = (
                    self._constraint[itr_comp] * phis_comp[itr_comp] * masks
                )
            incomp += phis_comp[itr_comp]
        incomp *= masks
        return incomp


class SemiGrandCanonicalEnsemble(EnsembleBase):
    r"""Semi-grand canonical ensemble where composition or potentials are fixed."""

    def __init__(
        self,
        num_comp: int,
        is_canonical: bool | NDArray[bool],
        constraint: np.ndarray,
    ):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            is_canonical:
                Boolean array deciding whether a component has constrained total amount
                (``True``, treated canonical) or constrained chemical potentials
                (``False``, treated grand-canonical).
            constraint:
                Value of the constraint for each component. For canonical components
                this constraint is the mean concentration, whereas for grand-canonical
                components the constraint is the scaled activity
                :math:`l_i e^{l_i \mu_i}`.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)

        shape = (num_comp,)
        self._is_canonical = np.array(np.broadcast_to(is_canonical, shape), dtype=bool)
        self.constraint = np.array(np.broadcast_to(constraint, shape))
        self._check()

    def _check(self):
        """Internal consistency check"""
        if not np.isclose(self._constraint[self._is_canonical].sum(), 1.0):
            self._logger.warning(
                "The sum of canonical constraints exceeds 1. The iteration may never "
                "converge in an incompressible system."
            )

    @property
    def is_canonical(self) -> NDArray[bool]:
        r"""Boolean array marking canonical components."""
        return self._is_canonical

    @is_canonical.setter
    def is_canonical(self, is_canonical_new: NDArray[bool]):
        r"""Set constraints of components.

        Args:
            is_canonical_new:
                Updated list of components treated canonically.
        """
        shape = (self.num_comp,)
        self._is_canonical = np.array(np.broadcast_to(is_canonical_new, shape))
        self._check()

    @property
    def constraint(self) -> np.ndarray:
        r"""The constraints of the components."""
        return self._constraint

    @constraint.setter
    def constraint(self, constraint_new: np.ndarray):
        r"""Set constraints of components.

        Args:
            constraint_new:
                Updated constraints.
        """
        shape = (self.num_comp,)
        self._constraint = np.array(np.broadcast_to(constraint_new, shape))
        self._check()

    def set_chemical_potential(
        self, comp_id: int, mu: float, size: float = 1.0
    ) -> None:
        r"""Set scaled constraint from chemical potentials by volume.

        Args:
            comp_id:
                Id of the component that is affected.
            mu:
                The chemical potential by volume, :math:`\mu_i`.
            size:
                The relative molecule volume :math:`l_i = \nu_i/\nu` with respect to the
                volume of a reference molecule :math:`\nu`.
        """
        self.constraint[comp_id] = size * np.exp(size * mu)

    def _compiled_impl(self) -> SemiGrandCanonicalEnsembleCompiled:
        """Implementation of creating a compiled ensemble instance.

        This method overwrites the interface
        :meth:`~flory.ensemble.base.EnsembleBase._compiled_impl` in
        :class:`~flory.ensemble.base.EnsembleBase`.

        Returns:
            : Instance of :class:`SemiGrandCanonicalEnsembleCompiled`.
        """

        return SemiGrandCanonicalEnsembleCompiled(
            is_canonical=self._is_canonical, constraint=self._constraint
        )
