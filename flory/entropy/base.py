"""Module for a general entropic energy of mixture.

"""

from typing import Optional
import numpy as np
from ..commom import *


class EntropyBase:
    """Base class for a general entropic energy of mixture."""

    def __init__(self, num_comp: int):
        r"""
        Args:
            num_comp:
                Number of components :math:`N_\mathrm{c}`.
        """
        self.num_comp = num_comp

    def _compiled_impl(self, **kwargs) -> object:
        """Implementation of creating a compiled entropy instance (Interface).
        This interface is meant to be overridden in derived classes. See :meth:`compiled`
        for more information on the compiled entropy instance.
        """
        raise NotImplementedError

    def _energy_impl(self, phis: np.ndarray) -> np.ndarray:
        """Implementation of calculating entropic energy (Interface).
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

    def compiled(self, **kwargs_full) -> object:
        r"""Make a compiled entropy instance for :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.
        This function requires the implementation of :meth:`_compiled_impl`. The entropy
        instance is a compiled class, which must implement:

            - property :samp:`num_comp`, which reports the number of components
              :math:`N_\mathrm{c}`.
            - property :samp:`num_feat`, which reports the number of features
              :math:`N_\mathrm{s}`.
            - method :samp:`partition(phis_comp, omegas, Js)`, which calculates the
              Boltzmann factors of the components under the mean fields :samp:`omegas` and
              stores them in :samp:`phis_comp` as the unnormalized volume fractions of
              components. :samp:`Js` contains the volumes of all the compartments. This
              function must return the single molecule partition functions of the
              components, which is the sum of the Boltzmann factors weighted by the
              volumes of compartments. Note that :samp:`omegas` contains the mean fields
              for the features instead of components. For example when we have
              polydispersity of the molecule sizes, all the polydispersed components share
              the same external fields, but they have their individual volume fractions.
              This function acts as the mapping from feature-based description of the
              system to the component-based description of the system. 
            - method :samp:`comp_to_feat(phis_feat, phis_comp)`, which maps the volume
              fractions of components :samp:`phis_comp` to the volume fractions of
              features :samp:`phis_feat`. Note that this method should modify
              :samp:`phis_feat` directly.
            - method :samp:`volume_derivative(phis_comp)`, which calculates the derivative
              of partition function part of the free energy with respect to the volumes of
              the compartments. In most of the cases, this is equivalent to the negative
              sum of the volume fractions of the components divided by the relative
              volumes of the molecules.

        Note that the compiled entropy instance acts as the bridge between the
        component-based description and the feature-based description of the systems
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
