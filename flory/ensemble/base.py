from ..commom import *


class EnsembleBase:
    def __init__(self, num_comp: int):
        r"""Base class for a general ensemble of mixture=.

        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`.
        """
        self.num_comp = num_comp

    def _compiled_impl(self, **kwargs) -> object:
        """returns ensemble instance containing necessary member functions for iteration."""
        raise NotImplementedError

    def compiled(self, **kwargs_full) -> object:
        """Create the ensemble instance containing necessary member functions for iteration
        This function requires the implementation of :meth:`_compiled_impl`. The
        ensemble instance is a compiled class, which must implement following compiled
        functions:

        Detailed documentation required here.

        Args:
            kwargs_full:
                The keyword arguments which allow additional unused arguments.

        Returns:
            : The compiler ensemble instance.
        """
        kwargs = filter_kwargs(kwargs_full, self._compiled_impl)
        return self._compiled_impl(**kwargs)
