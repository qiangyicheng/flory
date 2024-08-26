"""Module for a general ensemble of mixture.

"""

from ..commom import *


class EnsembleBase:
    """Base class for a general ensemble of mixture."""

    def __init__(self, num_comp: int):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`.
        """
        self.num_comp = num_comp

    def _compiled_impl(self, **kwargs) -> object:
        """Implementation of creating a compiled ensemble instance (Interface).
        This interface is meant to be overridden in derived classes. See :meth:`compiled`
        for more information on the compiled ensemble instance.
        """
        raise NotImplementedError

    def compiled(self, **kwargs_full) -> object:
        r"""Make a compiled ensemble instance for :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.
        This function requires the implementation of :meth:`_compiled_impl`. The ensemble
        instance is a compiled class, which must implement:

            - property :samp:`num_comp`, which reports the number of components
              :math:`N_\mathrm{c}`.
            - method :samp:`normalize(phis_comp, Qs, masks)`, which normalizes the
              Boltzmann factor stored in :samp:`phis_comp` into volume fractions of all
              components :math:`\phi_i^{(m)}` and save it back to :samp:`phis_comp`,
              making use of the single molecule partition function in :samp:`Qs`. The
              leading dimension of :samp:`phis_comp` and :samp:`Qs` must be both
              :samp:`num_comp`. :samp:`masks` is array with the same size as
              :samp:`phis_comp[0]`, using 0/1 to mark whether the compartment is alive.
              This method must report the incompressibility :math:`\sum_i
              \phi_i^{(m)} -1`. Note that this function is only aware of the number
              of components :math:`N_\mathrm{c}`. Mapping from/to features are handled by
              :mod:`~flory.entropy`.

        See :class:`~flory.ensemble.canonical.CanonicalEnsembleCompiled` for an example.

        Args:
            kwargs_full:
                The keyword arguments for :meth:`_compiled_impl` but allowing redundant
                arguments.

        Returns:
            : The compiled ensemble instance.
        """
        kwargs = filter_kwargs(kwargs_full, self._compiled_impl)
        return self._compiled_impl(**kwargs)
