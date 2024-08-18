import numpy as np
from ..commom import *

class InteractionBase:
    def __init__(self, num_comp:int):
        r"""Base class for a general interaction energy of mixture.

        Args:
            num_comp: 
                Number of components :math:`N_\mathrm{c}`.
        """
        self.num_comp = num_comp
        
    def _compiled_impl(self, **kwargs) -> object:
        """returns interaction instance containing necessary member functions for iteration.
        """
        raise NotImplementedError
    
    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        """returns interaction energy for a given composition"""
        raise NotImplementedError

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""returns full Jacobian :math:`\partial f/\partial \phi` for the given composition"""
        raise NotImplementedError

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""returns full Hessian :math:`\partial^2 f/\partial \phi^2` for the given composition"""
        raise NotImplementedError

    def compiled(self, **kwargs_full) ->object:
        """Create the interaction instance containing necessary member functions for iteration
        This function requires the implementation of :meth:`_interaction_impl`. The
        interaction instance is a compiled class, which must implement following compiled
        functions:
        
        Detailed documentation required here.
        
        Args:
            kwargs_full:
                The keyword arguments which allow additional unused arguments.

        Returns:
            : The compiler interaction instance.
        """
        kwargs = filter_kwargs(kwargs_full, self._compiled_impl)
        return self._compiled_impl(**kwargs)