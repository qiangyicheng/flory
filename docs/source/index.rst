.. flory documentation master file

'flory' python package
======================

The :mod:`flory` python package provides tools for investigating phase separation in mixtures based on Flory-Huggins theory.
Currently :mod:`flory` Contains:

Module :mod:`~flory.mcmp`
*******************************

Module :mod:`~flory.mcmp` finds multiple coexisting phases of incompressible multicomponent mixtures in canonical ensemble, whose average free energy density is in the form

.. math::
   \bar{f}({N_\mathrm{p}}, \{J_\alpha\}, \{\phi_i^{(\alpha)}\}) = \sum_{\alpha=1}^{{N_\mathrm{p}}} J_\alpha f(\{\phi_i^{(\alpha)}\}) \; ,

where :math:`N_\mathrm{c}` is the number of components, :math:`N_\mathrm{p}` is the number of phases, :math:`J_\alpha` is the volume fraction of the phase :math:`\alpha` and :math:`\phi_i^{(\alpha)}` is the volume fraction of the component :math:`i` in the phase :math:`\alpha`.
Each phase is considered to be homogeneous, a free energy density

.. math::
   f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{c}} \frac{\phi_i}{l_i} \ln \phi_i \; ,

where :math:`\chi_{ij}` is the Flory-Huggins interaction between component :math:`i` and :math:`j`, and :math:`l_i` is the relative molecule size of the component :math:`i`.
With given interaction matrix :math:`\chi_{ij}`, average volume fractions of all components across the system :math:`\bar{\phi}_i` and the relative molecule sizes :math:`l_i`, module :mod:`~flory.mcmp` provides tools to find the coexisting phases in equilibrium.

For example:

.. literalinclude:: /../../examples/find_coexiting_phases.py
   :linenos:
   :lines: 1,3,4,6

See :ref:`examples` for more use cases.

Contents
==================

.. toctree::
   :maxdepth: 1
   
   examples
   theory
   api
   detail

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
