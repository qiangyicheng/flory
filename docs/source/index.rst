.. flory documentation master file

'flory' Python package
======================

The :mod:`flory` Python package provides tools for investigating phase separation in
multicomponent mixtures. In particular, it allows to determine equilibrium states of
:math:`N_\mathrm{p}` coexisting phases, each described by the volume fractions
:math:`\phi_i^{(\alpha)}` of the :math:`i=1, \ldots, N_\mathrm{c}` components.

We currently focus on incompressible systems with constant volume (cannonical ensemble),
whose average free energy density is given by 

.. math::
   \bar{f}({N_\mathrm{p}}, \{J_\alpha\}, \{\phi_i^{(\alpha)}\}) = \sum_{\alpha=1}^{{N_\mathrm{p}}} J_\alpha f(\{\phi_i^{(\alpha)}\}) \; ,

where :math:`J_\alpha` is the fraction of volume occupied by phase :math:`\alpha`.
Each phase is described by a Flory-Huggins free energy

.. math::
   f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{c}} \frac{\phi_i}{l_i} \ln \phi_i \; ,

where :math:`\chi_{ij}` is the Flory interaction parameter between component :math:`i`
and :math:`j`, and :math:`l_i` is the relative molecule size of the component :math:`i`.
For a given interaction matrix :math:`\chi_{ij}`, average volume fractions of all
components across the system :math:`\bar{\phi}_i`, and the relative molecule sizes
:math:`l_i`, the coexisting phases are those configurations that minimize :math:`\bar f`,
which is used by :mod:`flory`:

.. literalinclude:: /../../examples/find_coexiting_phases.py
   :linenos:
   :lines: 1,3,4,6

The details are described in :func:`~flory.mcmp.find_coexisting_phases`, and
:ref:`examples` lists more use cases.

Contents
==================

.. toctree::
   :maxdepth: 1

   self
   examples
   theory
   api
   detail

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
