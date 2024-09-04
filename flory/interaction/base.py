"""Module for a general interaction energy of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""
from __future__ import annotations

import numpy as np

from ..common import *


class InteractionBaseCompiled:
    r"""Abstract base class for a general compiled interaction.

    This abstract class defines the necessary members of a compiled constraint instance.
    This abstract class does not inherit from :class:`abc.ABC`, since the
    :func:`numba.experimental.jitclass` currently does not support some members of
    :class:`abc.ABC`. A compiled class derived from :class:`InteractionBaseCompiled` is in
    general stateless. In other words, the compiled interaction instance never managers its
    own data. Note that the methods may change the input arrays inplace to avoid creating
    them each time.
    """

    @property
    def num_feat(self) -> int:
        r"""Number of features :math:`N_\mathrm{S}`."""

        raise NotImplementedError

    def volume_derivative(
        self, potential: np.ndarray, phis_feat: np.ndarray
    ) -> np.ndarray:
        r"""Calculate the volume derivatives of interaction energy.

        This method calculates the partial derivative of interaction part of the free
        energy with respect to the volumes of the compartments :math:`\partial
        f_\mathrm{interaction}/\partial J_m`. In most of the cases, this is just the
        interaction energy density in all compartments.

        Args:
            potential:
                Constant. 2D array with the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`,
                containing the part of the field :math:`w_r^{(m)}` contributed by the
                interaction. Usually this is the returned value of :meth:`potential`. This
                parameter is passed in since usually the calculation of interaction energy
                density can be accelerated by directly using the potential.
            phis_feat:
                Constant. The 2D array with the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`,
                containing the volume fractions of features :math:`\phi_r^{(m)}`.

        Returns:
            : The volume derivatives.
        """
        raise NotImplementedError

    def potential(self, phis_feat: np.ndarray) -> np.ndarray:
        r"""Calculate part of :math:`w_r^{(m)}` from interaction.

        This method calculates the part of mean field :math:`w_r^{(m)}` contributed by the
        interaction. Usually this is just the Jacobian of the interaction energy with
        respect to the volume fractions of features in each compartment. This method
        should return the result directly.

        Args:
            phis_feat:
                Constant. The 2D array with the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`,
                containing the volume fractions of features :math:`\phi_r^{(m)}`.

        Returns:
            : Part of :math:`w_r^{(m)}` contributed by the interaction.
        """
        raise NotImplementedError

    def incomp_coef(self, phis_feat: np.ndarray) -> float | np.ndarray:
        r"""Calculate the coefficient for incompressibility.

        This method calculates the coefficient for incompressibility during iteration.
        This coefficient is derived heuristically. The most common way is to partially
        make use of the incompressibility in the expression of :meth:`potential`, and then
        determine the changes of :meth:`potential` after applying incompressibility. The
        coefficient can be compartment-dependent. This method should return the result
        directly.

        Args:
            phis_feat:
                Constant. The 2D array with the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`,
                containing the volume fractions of features :math:`\phi_r^{(m)}`.

        Returns:
            : The coefficient for incompressibility.
        """
        raise NotImplementedError


class InteractionBase:
    r"""Base class for a general interaction energy of mixture.

    The class :class:`InteractionBase` is designed to use number of components
    :math:`N_\mathrm{C}` since this is the more physically comprehensive way to describe a
    mixture, even though there might be redundancies in such definition. For example, in a
    incompressible canonical mixture of polydispersed components, the system can be
    uniquely described by volume fractions of features, where one feature can contain
    multiple components with different molecule sizes. In such cases the interaction be
    expressed by the interaction between features, giving significant simplification of
    the numerics. In other words, there exists degeneracy of components. However, the
    class :class:`InteractionBase` does not include this directly. Instead, such
    system-specific things are considered by the compiled interaction classes, which
    should only be aware of the feature-based description. see
    :class:`~flory.interaction.flory_huggins.FloryHugginsInteractionCompiled` for an
    example.
    """

    def __init__(self, num_comp: int):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
        """
        self.num_comp = num_comp

    def _compiled_impl(self, **kwargs) -> InteractionBaseCompiled:
        r"""Implementation of creating a compiled interaction instance (Interface).

        This interface is meant to be overridden in derived classes. See :meth:`compiled`
        for more information on the compiled interaction instance.
        """
        raise NotImplementedError

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating interaction energy :math:`f_\mathrm{interaction}` (Interface).

        This interface is meant to be overridden in derived classes. Multiple compositions
        should be allowed. This method is not necessary for the core algorithm.
        
        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
                
        Returns:
            : The entropic energy density.
        """
        raise NotImplementedError

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Jacobian :math:`\partial f_\mathrm{interaction}/\partial \phi_i` (Interface).

        This interface is meant to be overridden in derived classes. Multiple compositions
        should be allowed. This method is not necessary for the core algorithm.
        
        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
                
        Returns:
            : The full Jacobian.
        """
        raise NotImplementedError

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Hessian :math:`\partial^2 f_\mathrm{interaction}/\partial \phi_i^2` (Interface).

        This interface is meant to be overridden in derived classes. Multiple compositions
        should be allowed. This method is not necessary for the core algorithm.
        
        Args:
            phis:
                The volume fractions of the phase(s) :math:`\phi_{p,i}`. if multiple
                phases are included, the index of the components must be the last
                dimension.
                
        Returns:
            : The full Hessian.
        """
        raise NotImplementedError

    def compiled(self, **kwargs_full) -> InteractionBaseCompiled:
        r"""Make a compiled interaction instance for :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.

        This function requires the implementation of :meth:`_compiled_impl`. The
        interaction instance is a compiled class, which must implement a list of methods
        or properties. See :class:`InteractionBaseCompiled` for the list and the detailed
        information. See also
        :class:`~flory.ensemble.entropy.FloryHugginsInteractionCompiled` for an example.
        Note that different from the class :class:`InteractionBase` itself, the returned
        compiled class use the feature-based description, and can consider the degeneracy
        of components. 

        Args:
            kwargs_full:
                The keyword arguments for :meth:`_compiled_impl` but allowing redundant
                arguments.

        Returns:
            : The compiler interaction instance.
        """
        kwargs = filter_kwargs(kwargs_full, self._compiled_impl)
        return self._compiled_impl(**kwargs)
