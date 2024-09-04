"""Module for a general ensemble of mixture.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import numpy as np

from ..common import filter_kwargs


class EnsembleBaseCompiled:
    r"""Abstract base class for a general compiled ensemble.

    This abstract class defines the necessary members of a compiled constraint instance.
    This abstract class does not inherit from :class:`abc.ABC`, since the
    :func:`numba.experimental.jitclass` currently does not support some members of
    :class:`abc.ABC`. A compiled class derived from :class:`EnsembleBaseCompiled` is in
    general stateless. In other words, the compiled ensemble instance never managers its
    own data. Note that the methods may change the input arrays inplace to avoid creating
    them each time.
    """

    @property
    def num_comp(self) -> int:
        r"""Number of components :math:`N_\mathrm{C}`."""
        raise NotImplementedError

    def normalize(
        self, phis_comp: np.ndarray, Qs: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        r"""Normalize the volume fractions of components.

        This method normalizes the Boltzmann factor stored in :paramref:`phis_comp` into
        volume fractions of all components :math:`\phi_i^{(m)}` and save it back to
        :paramref:`phis_comp`, making use of the single molecule partition function in
        :paramref:`Qs`. The exact form of such normalization depends on the emsemble. This
        method must report the incompressibility :math:`\sum_i \phi_i^{(m)} -1`. Note that
        this function is only aware of the number of components :math:`N_\mathrm{C}`.
        Mapping from/to features are handled by :mod:`~flory.entropy`.

        Args:
            phis_comp:
                Mutable. The 2D array with the size of :math:`N_\mathrm{C} \times N_\mathrm{M}`, containing
                Boltzmann factors of the components, which are proportional
                to resulting volume fractions.
            Qs:
                Constant. The 1D array with the size of :math:`N_\mathrm{C}`, containing
                single molecule partition functions of the components.
            masks:
                Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the masks to
                mark whether the compartment is living or not.

        Returns:
            : The incompressibility.
        """
        raise NotImplementedError


class EnsembleBase:
    """Base class for a general ensemble of mixture."""

    def __init__(self, num_comp: int):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{C}`.
        """
        self.num_comp = num_comp

    def _compiled_impl(self, **kwargs) -> EnsembleBaseCompiled:
        """Implementation of creating a compiled ensemble instance (Interface).

        This interface is meant to be overridden in derived classes. See :meth:`compiled`
        for more information on the compiled ensemble instance.
        """
        raise NotImplementedError

    def compiled(self, **kwargs_full) -> EnsembleBaseCompiled:
        r"""Make a compiled ensemble instance for :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.

        This function requires the implementation of :meth:`_compiled_impl`. The ensemble
        instance is a compiled class, which must implement a list of methods or
        properties. See :class:`EnsembleBaseCompiled` for the list and the detailed
        information. Also see :class:`~flory.ensemble.canonical.CanonicalEnsembleCompiled`
        for an example.

        Args:
            kwargs_full:
                The keyword arguments for :meth:`_compiled_impl` but allowing redundant
                arguments.

        Returns:
            : The compiled ensemble instance.
        """
        kwargs = filter_kwargs(kwargs_full, self._compiled_impl)
        return self._compiled_impl(**kwargs)
