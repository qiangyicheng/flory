"""Module for reactive canonical ensemble of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np
from numba import float64, int32
from numba.experimental import jitclass

from .base import EnsembleBase, EnsembleBaseCompiled


@jitclass(
    [
        ("_num_comp", int32),  # a scalar
        ("_weights", float64[:, :]),
        ("_weights_inv", float64[:, :]),
        ("_targets", float64[::1]),  # a C-continuous array
    ]
)
class ReactiveCanonicalEnsembleCompiled(EnsembleBaseCompiled):
    r"""Compiled class for canonical ensemble with conserved combinations of fractions.

    In contrast to the canonical ensemble, where the average volume fractions of the
    components are fixed, we here only fix a linear combination of the fractions, which
    correspond to conserved quantities :math:`\bar\psi_\beta` of reactions,

        .. math::
            \bar\psi_\beta = \sum_i B_{\beta,i} \bar\phi_i
            = \frac{\sum_m J_m \sum_i B_{\beta,i} \phi_i^{(m)} }{\sum_m J_m}

    Idea: This constraint could be implemented by guessing `phi_means` and using a
    projection method to enforce the linear constraint.
    """

    def __init__(
        self, weights: np.ndarray, targets: np.ndarray, *, weights_inv: np.ndarray
    ):
        r"""
        Args:
            weights:
                The full affine constraint matrix including the incompressibility row.
            targets:
                The associated targets, including the value 1 for incompressibility.
            weights_inv:
                The pseudo-inverse of the full constraint matrix.
        """
        self._num_comp = weights.shape[1]
        self._weights = weights
        self._weights_inv = weights_inv
        self._targets = targets

    @property
    def num_comp(self):
        return self._num_comp

    def normalize(
        self, phis_comp: np.ndarray, Qs: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        r"""Normalize component fractions under conserved linear combinations.

        The equilibrium mean fractions under linear conservation laws are the constrained
        Gibbs distribution

            .. math::
                \bar\phi_i \propto Q_i \exp\left(-\sum_\beta \lambda_\beta B_{\beta,i}\right),

        where the Lagrange multipliers :math:`\lambda_\beta` are chosen such that the
        conserved linear combinations match their targets. This is the physically correct
        analogue of the canonical ensemble for the reaction-constrained case and ensures
        equal chemical potentials for components that are only linked through a conserved
        sum.

        Args:
            phis_comp:
                Mutable component Boltzmann factors, updated in-place to normalized
                fractions.
            Qs:
                Single molecule partition functions of components.
            masks:
                Masks indicating whether compartments are active.

        Returns:
            The incompressibility in each compartment.
        """
        # Solve the constrained Gibbs distribution for the mean fractions using a damped
        # Newton update in the dual variables λ. The equilibrium condition is
        #   phi_i ∝ Q_i exp(-(B^T λ)_i)
        # with the reaction constraints B @ phi = target. This yields the same chemical
        # potential for components that are only coupled through a conserved reaction sum.
        #
        # The full Newton step can be too aggressive for reaction matrices that are nearly
        # singular or poorly scaled. We therefore accept only steps that reduce the
        # residual norm; otherwise the step is backtracked until it becomes stable. This
        # keeps the update in the admissible Gibbs manifold while still converging to the
        # correct constrained solution for the canonical limit and the reaction-constrained
        # case.
        multipliers = np.zeros(self._weights.shape[0])
        phi_means = Qs / np.maximum(Qs.sum(), 1e-30)
        for _ in range(200):
            exp_term = np.exp(-(self._weights.T @ multipliers))
            phi_means = Qs * exp_term
            phi_means /= np.maximum(phi_means.sum(), 1e-30)

            residue = self._weights @ phi_means - self._targets
            res_norm = np.max(np.abs(residue))
            if res_norm < 1e-12:
                break

            covariance = np.diag(phi_means) - np.outer(phi_means, phi_means)
            jacobian = self._weights @ covariance @ self._weights.T
            jacobian += 1e-12 * np.eye(self._weights.shape[0])

            # The reaction residual is r(λ) = B @ φ(λ) - target. Its Jacobian is
            # J = B @ Σ @ B^T with Σ = diag(φ) - φφ^T, and the Newton step is
            # J * δλ = r. The sign is therefore positive here; the backtracking loop
            # below then keeps the step stable whenever the raw Newton update is too
            # aggressive or the Jacobian is nearly singular.
            step = np.linalg.solve(jacobian, residue)
            step_scale = 1.0
            accepted = False
            while step_scale > 1e-8:
                trial_multipliers = multipliers + step_scale * step
                trial_exp = np.exp(-(self._weights.T @ trial_multipliers))
                trial_phi = Qs * trial_exp
                trial_phi /= np.maximum(trial_phi.sum(), 1e-30)
                trial_residue = self._weights @ trial_phi - self._targets
                trial_res_norm = np.max(np.abs(trial_residue))
                if trial_res_norm < res_norm:
                    multipliers = trial_multipliers
                    accepted = True
                    break
                step_scale *= 0.5

            if not accepted:
                break

        # enforce individual average fractions
        incomp = -1.0 * np.ones_like(phis_comp[0])
        for itr_comp in range(self._num_comp):
            factor = phi_means[itr_comp] / Qs[itr_comp]
            phis_comp[itr_comp] = factor * phis_comp[itr_comp] * masks
            incomp += phis_comp[itr_comp]
        incomp *= masks
        return incomp


class ReactiveCanonicalEnsemble(EnsembleBase):
    r"""Class for an canonical ensemble with conserved linear combinations of fractions.

    The particular form of the conservation law reads

    .. math::
            \bar\psi_\beta = \sum_i B_{\beta,i} \bar\phi_i
            = \frac{\sum_m J_m \sum_i B_{\beta,i} \phi_i^{(m)} }{\sum_m J_m}

    This reduces to the canonical ensemble for :math:`B_{\beta,i} = \delta_{\beta,i}`,
    where the average fractions are constrained, :math:`\bar\psi_\beta = \bar\phi_i`.
    In other cases, the constraints can capture chemical reactions, where only the total
    particles counts are conserved.
    """

    def __init__(self, num_comp: int, weights: np.ndarray, targets: np.ndarray):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
            weights:
                The weights :math:`B_{\beta,i}` with which each component contributes to
                the constraints. A single constraint may be passed as a 1D array.
            targets:
                The average value of the constraint :math:`\bar\psi_\beta`.
        """
        super().__init__(num_comp)
        self._logger = logging.getLogger(self.__class__.__name__)

        self._weights = np.asarray(weights, dtype=float)
        if self._weights.ndim == 1:
            self._weights = self._weights[np.newaxis, :]
        if self._weights.ndim != 2:
            raise ValueError("Weights must be 1D or 2D array")
        if self._weights.shape[1] != self.num_comp:
            raise ValueError("Second dimension of `weights` must equal component count")
        self.targets = targets

    @property
    def weights(self) -> np.ndarray:
        r"""The weights :math:`B_{\beta,i}` for each component."""
        return self._weights

    @property
    def targets(self) -> np.ndarray:
        r"""The average value of the constraint :math:`\bar\psi_\beta`."""
        return self._targets

    @targets.setter
    def targets(self, targets_new: np.ndarray):
        r"""Set the average values of the constraints.

        Args:
            targets_new:
                Updated average value of the constraint :math:`\bar\psi_\beta`.
        """
        targets_new = np.asarray(targets_new, dtype=float, copy=True)  # copy data
        if targets_new.ndim == 0:
            targets_new = targets_new.reshape(1)
        if targets_new.size != self._weights.shape[0]:
            raise ValueError(
                "The number of target values must equal the number of reaction rows in `weights`."
            )
        self._targets = targets_new

    def _compiled_impl(self) -> ReactiveCanonicalEnsembleCompiled:
        """Implementation of creating a compiled ensemble instance.

        This method overwrites the interface
        :meth:`~flory.ensemble.base.EnsembleBase._compiled_impl` in
        :class:`~flory.ensemble.base.EnsembleBase`.

        Returns:
            : Instance of :class:`CanonicalEnsembleCompiled`.
        """
        return ReactiveCanonicalEnsembleCompiled(
            self.weights,
            self.targets,
            weights_inv=np.linalg.pinv(self.weights),
        )
