"""Module for a general constraint.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np

from ..common import filter_kwargs


class ConstraintBaseCompiled:
    r"""Abstract base class for a general compiled constraint.

    This abstract class defines the necessary members of a compiled constraint instance.
    This abstract class does not inherit from :class:`abc.ABC`, since the
    :func:`numba.experimental.jitclass` currently does not support some members of
    :class:`abc.ABC`. Due to the variety of constraints, a compiled class derived from
    :class:`ConstraintBaseCompiled` is in general stateful. In other words, the compiled
    constraint instance must manage its own data such as the Lagrange multiplier.
    Besides, it can also keep other data to avoid repeating certain calculation.
    Therefore, the class :class:`~flory.mcmp.finder.CoexistingPhasesFinder` uses the
    constraint instances in the manner of a prepare-access-evolve triplet.
    """

    @property
    def num_feat(self) -> int:
        r"""Number of features :math:`N_\mathrm{S}`."""
        raise NotImplementedError

    @property
    def potential(self) -> np.ndarray:
        r"""The potential for features generated by the constraint.

        This property typically contains the Jacobian of the constraint part of the free
        energy with respect to the volume fractions of the features, which is an array
        with the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`. Note that this property should be
        used after :meth:`prepare` is called.
        """
        raise NotImplementedError

    @property
    def volume_derivative(self) -> np.ndarray:
        r"""The volume derivatives of the constraint part of entropic energy.

        This property typically contains the Jacobian of the constraint part of the free
        energy with respect to the volumes of the compartments, which is an array with the
        size of :math:`N_\mathrm{M}`. Note that this property should be used after :meth:`prepare` is
        called.
        """
        raise NotImplementedError

    def initialize(self, num_part: int) -> None:
        r"""Initialize the internal data of the constraint.

        Typically this function initialize the Lagrange multiplier according to the
        number of compartments.

        Args:
            num_part:
                Constant. Number of compartments :math:`N_\mathrm{M}`.
        """
        raise NotImplementedError

    def prepare(self, phis_feat: np.ndarray, Js: np.ndarray, masks: np.ndarray) -> None:
        r"""Prepare the constraint.

        This function prepares the constraint according to the volume fractions of
        features and the masks of the compartments. Usually this includes the calculation
        of :attr:`potential` and :attr:`volume_derivative`.


        Args:
            phis_feat:
                Constant. The 2D array with the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`,
                containing the volume fractions of features :math:`\phi_r^{(m)}`.
            Js:
                Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the relative
                volumes of compartments :math:`J_m`.
            masks:
                Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the masks to
                mark whether the compartment is living or not.
        """
        raise NotImplementedError

    def evolve(self, step: float, masks: np.ndarray) -> float:
        r"""Evolve the internal state of the constraint.

        Args:
            step:
                Constant. The step size of the evolution.
            masks:
                Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the masks to
                mark whether the compartment is living or not.

        Returns:
            : the max absolute residue.
        """

        raise NotImplementedError


class ConstraintBase:
    """Base class for a general constraint.

    A constraint can be either local or global. Local constraint means that it needs be
    satisfied in all compartments. For example, the charge balance is a local constraint.
    Global constraint means that it only needs to be satisfied by the mean value. For
    example, the volume conservation is a global constraint. Note that we expect that
    constraints always act on feature, in stead of components.
    """

    def __init__(self, num_feat: int):
        r"""
        Args:
            num_feat:
                Number of features :math:`N_\mathrm{S}`.
        """
        self.num_feat = num_feat

    def _compiled_impl(self, **kwargs) -> ConstraintBaseCompiled:
        """Implementation of creating a compiled constraint instance (Interface).

        This interface is meant to be overridden in derived classes. See :meth:`compiled`
        for more information on the compiled ensemble instance.
        """
        raise NotImplementedError

    def compiled(self, **kwargs_full) -> ConstraintBaseCompiled:
        r"""Make a compiled constraint instance for :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.

        This function requires the implementation of :meth:`_compiled_impl`. The
        constraint instance is a compiled class, which must implement a list of methods or
        properties. See :class:`ConstraintBaseCompiled` for the list and the detailed
        information. Also see
        :class:`~flory.constraint.canonical.CanonicalEnsembleCompiled` for an example of
        the implementation.

        Args:
            kwargs_full:
                The keyword arguments for :meth:`_compiled_impl` but allowing redundant
                arguments.

        Returns:
            : The compiled constraint instance.
        """
        kwargs = filter_kwargs(kwargs_full, self._compiled_impl)
        return self._compiled_impl(**kwargs)
