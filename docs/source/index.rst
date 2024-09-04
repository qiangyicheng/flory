.. flory documentation master file

'flory' Python package
======================

The :mod:`flory` Python package provides tools for investigating phase separation in
multicomponent mixtures. In particular, it allows to determine equilibrium states of
:math:`N_\mathrm{P}` coexisting phases, each described by the volume fractions
:math:`\phi_{p,i}` of the :math:`i=1, \ldots, N_\mathrm{C}` components.

:mod:`flory` finds coexisting phases by minimizing the average free energy density

.. math::
    \bar{f}({N_\mathrm{P}}, \{J_p\}, \{\phi_{p,i}\}) = \sum_{p=1}^{{N_\mathrm{P}}} J_p f(\{\phi_{p,i}\}) \; ,

where :math:`J_p` is the fraction of the system volume occupied by phase :math:`p`.


:mod:`flory` supports different forms of interaction, entropy, ensemble, and constraints to
assemble the free energy of the phases. For example, with the commonly used Flory-Huggins
free energy, the free energy density of each homogeneous phase reads

.. math::
   f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{C}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{C}} \frac{\phi_i}{l_i} \ln \phi_i \; ,

where :math:`\chi_{ij}` is the Flory interaction parameter between component :math:`i`
and :math:`j`, and :math:`l_i` is the relative molecule size of the component :math:`i`.
For a given interaction matrix :math:`\chi_{ij}`, average volume fractions of all
components across the system :math:`\bar{\phi}_i`, and the relative molecule sizes
:math:`l_i`, the coexisting phases are those configurations that minimize :math:`\bar f`,
which is used by :mod:`flory`:

.. literalinclude:: /../../examples/find_coexiting_phases.py
   :linenos:
   :lines: 1,3,4,6

The example above is equivalent to a detailed one, 

.. literalinclude:: /../../examples/find_coexiting_phases_class.py
   :linenos:
   :lines: 6-8, 10

See :ref:`examples` for more use cases.

Contents
==================

.. toctree::
   :maxdepth: 1

   self
   examples
   theory
   symbol
   api
   detail

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
