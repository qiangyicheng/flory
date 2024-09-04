"""Module for a general entropic energy of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np

from ..common import filter_kwargs


class EntropyBaseCompiled:
    r"""Abstract base class for a general compiled entropy.

    This abstract class defines the necessary members of a compiled constraint instance.
    This abstract class does not inherit from :class:`abc.ABC`, since the
    :func:`numba.experimental.jitclass` currently does not support some members of
    :class:`abc.ABC`. A compiled class derived from :class:`EntropyBaseCompiled` is in
    general stateless. In other words, the compiled entropy instance never managers its
    own data. Note that the methods may change the input arrays inplace to avoid creating
    them each time.

    Class derived from :class:`EntropyBaseCompiled` is of important and special role in
    the core numerical method of the package :mod:`flory`. In principle, many
    complications of multicomponent systems are expressed by the entropy instance, ranging
    from simple molecule sizes, polydispersity, to even copolymers. In particular, the
    entropy instance acts as the bridge between the component-based description and the
    feature-based description. So for the core algorithm, entropy instance is the only
    instance that is aware both descriptions.

    """

    @property
    def num_comp(self) -> int:
        r"""Number of components :math:`N_\mathrm{C}`."""
        raise NotImplementedError

    @property
    def num_feat(self) -> int:
        r"""Number of features :math:`N_\mathrm{S}`."""
        raise NotImplementedError

    def partition(
        self, phis_comp: np.ndarray, omegas: np.ndarray, Js: np.ndarray
    ) -> np.ndarray:
        r"""Calculate the partition function and Boltzmann factors.

        This method calculates the Boltzmann factors :math:`p_i^{(m)}` of the components
        under the mean fields :paramref:`omegas` and stores them in :paramref:`phis_comp`
        as the volume fractions of components before normalization. This method must
        return the single molecule partition functions of the components :math:`Q_i`,
        which is the sum of the Boltzmann factors weighted by the volumes of compartments
        :math:`J_m` in :paramref:`Js`. Note that :paramref:`omegas` contains the mean
        fields for the features :math:`w_r^{(m)}` instead of components. For example when
        we have polydispersity of the molecule sizes, all the polydispersed components of
        the same kind share the same external fields, but with different volume fractions.
        In such cases the number of features :math:`N_\mathrm{S}` can be much smaller than
        number of components :math:`N_\mathrm{C}`. This method acts as the mapping from
        feature-based description of the system to the component-based description of the
        system. Note that this method should modify :paramref:`phis_comp` directly.

        Args:
            phis_comp:
                Output. The 2D array with the size of :math:`N_\mathrm{C} \times N_\mathrm{M}`,
                containing Boltzmann factors :math:`p_i^{(m)}`, namely the volume
                fractions of components before normalization.
            omegas:
                Constant. The 2D array with the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`,
                containing the mean field felt by the features :math:`\phi_r^{(m)}`.
            Js:
                Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the relative
                volumes of compartments :math:`J_m`.

        Returns:
            : The single molecule partition functions of components :math:`Q_i`.
        """
        raise NotImplementedError

    def comp_to_feat(self, phis_feat: np.ndarray, phis_comp: np.ndarray) -> None:
        r"""Convert the fractions of components into features.

        This method converts the volume fractions of components :math:`\phi_i^{(m)}` in
        :paramref:`phis_comp` to the volume fractions of features :math:`\phi_r^{(m)}` in
        :paramref:`phis_feat`. Note that this method should modify :paramref:`phis_feat`
        directly.

        Args:
            phis_feat:
                Output. The 2D array with the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`,
                containing the volume fractions of features :math:`\phi_r^{(m)}`.
            phis_comp:
                Constant. The 2D array with the size of :math:`N_\mathrm{C} \times N_\mathrm{M}`,
                containing the volume fractions of components :math:`\phi_i^{(m)}`.

        """
        raise NotImplementedError

    def volume_derivative(self, phis_comp: np.ndarray) -> np.ndarray:
        r"""Calculate the volume derivatives of entropic energy.

        This method calculates the partial derivative of partition function part of the
        free energy with respect to the volumes of the compartments :math:`\partial
        f_\mathrm{entropy}/\partial J_m`.

        In most of the cases, this is equivalent to the negative sum of the volume
        fractions of the components :math:`\phi_i^{(m)}` divided by the relative volumes
        of the molecules :math:`l_i`. Note that this module is separated from the
        iteration algorithm to remove from it any explicit dependence of the molecule
        parameters such as :math:`l_i`.

        Returns:
            : The volume derivatives.
        """
        raise NotImplementedError


class EntropyBase:
    """Base class for a general entropic energy of mixture."""

    def __init__(self, num_comp: int):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
        """
        self.num_comp = num_comp

    def _compiled_impl(self, **kwargs) -> EntropyBaseCompiled:
        """Implementation of creating a compiled entropy instance (Interface).

        This interface is meant to be overridden in derived classes. See :meth:`compiled`
        for more information on the compiled entropy instance.
        """
        raise NotImplementedError

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating entropic energy :math:`f_\mathrm{entropy}` (Interface).

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
        r"""Implementation of calculating Jacobian :math:`\partial f_\mathrm{entropy}/\partial \phi_i` (Interface).

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
        r"""Implementation of calculating Hessian :math:`\partial^2 f_\mathrm{entropy}/\partial \phi_i^2` (Interface).

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

    def compiled(self, **kwargs_full) -> EntropyBaseCompiled:
        r"""Make a compiled entropy instance for :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.

        This function requires the implementation of :meth:`_compiled_impl`. The entropy
        instance is a compiled class, which must implement a list of methods or
        properties. See :class:`EntropyBaseCompiled` for the list and the detailed
        information.

        Note that the compiled entropy instance acts as the bridge between the
        component-based description and the feature-based description of the system's
        states. In contrast, compiled classes in :mod:`~flory.ensemble` should be only
        aware of component-based description, compiled classes in
        :mod:`~flory.interaction` and :mod:`~flory.constraint` should be only aware of
        feature-based description.

        See :class:`~flory.ensemble.entropy.IdealGasEntropyCompiled` for an example.

        Args:
            kwargs_full:
                The keyword arguments for :meth:`_compiled_impl` but allowing redundant
                arguments.

        Returns:
            : The compiled entropy instance.

        """

        kwargs = filter_kwargs(kwargs_full, self._compiled_impl)
        return self._compiled_impl(**kwargs)
