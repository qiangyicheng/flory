"""Module for a general interaction energy of mixture.

"""
import numpy as np
from ..commom import *

class InteractionBase:
    """Base class for a general interaction energy of mixture.
    The class :class:`InteractionBase` is designed to use number of components
    :math:`N_\mathrm{c}` since this is the more physically comprehensive way to describe a
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
    def __init__(self, num_comp:int):
        r"""
        Args:
            num_comp: 
                Number of components :math:`N_\mathrm{c}`.
        """
        self.num_comp = num_comp
        
    def _compiled_impl(self, **kwargs) -> object:
        """Implementation of creating a compiled interaction instance (Interface).
        This interface is meant to be overridden in derived classes. See :meth:`compiled`
        for more information on the compiled interaction instance.
        """
        raise NotImplementedError
    
    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        """Implementation of calculating interaction energy (Interface).
        This interface is meant to be overridden in derived classes. Multiple compositions
        should be allowed.
        """
        raise NotImplementedError

    def _jacobian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Jacobian :math:`\partial f/\partial \phi` (Interface).
        This interface is meant to be overridden in derived classes. Multiple compositions
        should be allowed.
        """
        raise NotImplementedError

    def _hessian_impl(self, phis: np.ndarray) -> np.ndarray:
        r"""Implementation of calculating Hessian :math:`\partial^2 f/\partial \phi^2` (Interface).
        This interface is meant to be overridden in derived classes. Multiple compositions
        should be allowed.
        """
        raise NotImplementedError

    def compiled(self, **kwargs_full) ->object:
        r"""Make a compiled interaction instance for :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.
        This function requires the implementation of :meth:`_compiled_impl`. The interaction
        instance is a compiled class, which must implement:
        
            - property :samp:`num_feat`, which reports the number of features
              :math:`N_\mathrm{f}`.
            - method :samp:`energy(potential, phis_feat)`, which calculates the
              interaction energy from the volume fractions of the features. Note that the
              index of features is the first dimension of :samp:`phis_feat`, different
              from :meth:`_energy_impl`. This method should return the result directly.
            - method :samp:`potential(self, phis_feat)`, which calculates the potential
              felt by each feature from the volume fractions of the features. This is
              effectively the Jacobian of the interaction energy with respect to the
              volume fractions of features. Note that the index of features is the first
              dimension of :samp:`phis_feat`, different from :meth:`_jacobian_impl`.
              This method should return the result directly.
            - method :samp:`incomp_coef(self, phis_feat)`, which calculates the
              coefficient for incompressibility during calculation. This coefficient is
              derived heuristically. The most common way is to partially make use of the
              incompressibility in the expression of :samp:`potential`, and then determine
              the changes of :samp:`potential` after applying incompressibility. The
              coefficient can be compartment-dependent. This method should return the
              result directly.
        
        Note that different from the class :class:`InteractionBase` itself, the returned compiled class can consider the degeneracy of components.
        
        See :class:`~flory.ensemble.entropy.FloryHugginsInteractionCompiled` for an example.
        
        Args:
            kwargs_full:
                The keyword arguments which allow additional unused arguments.

        Returns:
            : The compiler interaction instance.
        """
        kwargs = filter_kwargs(kwargs_full, self._compiled_impl)
        return self._compiled_impl(**kwargs)