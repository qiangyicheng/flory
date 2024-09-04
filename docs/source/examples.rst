Examples
======================

Here we provide a few examples for using package.

Find coexisting phases once
---------------------------
Since we only need to find the coexisting phases once, we assign values to the :math:`\chi_{ij}` matrix and the average volume fractions :math:`\bar{\phi}_i` , and call the wrapper function :meth:`~flory.shortcut.find_coexisting_phases` directly.
Here we consider a symmetric two-component system, with :math:`\chi=4`:

.. literalinclude:: /../../examples/find_coexiting_phases.py
   :emphasize-lines: 6
   :linenos:

We obtain two symmetric phases:

.. literalinclude:: /../../examples/find_coexiting_phases.py.out
   :linenos:


By adding a :paramref:`~flory.mcmp.finder.CoexistingPhasesFinder.sizes` parameter, we can also investigate the coexisting phases where the components have different molecule sizes:

.. literalinclude:: /../../examples/find_coexiting_phases_unequal_sizes.py
   :emphasize-lines: 5, 7
   :linenos:

which gives two asymmetric phases:

.. literalinclude:: /../../examples/find_coexiting_phases_unequal_sizes.py.out
   :linenos:

Extension to more components is straight forward:

.. literalinclude:: /../../examples/find_coexiting_phases_ternary.py
   :emphasize-lines: 3-5
   :linenos:

which gives two phases:

.. literalinclude:: /../../examples/find_coexiting_phases_ternary.py.out
   :linenos:

Construct a 2D phase diagram
----------------------------
When constructing a phase diagram, we usually need to find coexisting phases for multiple instances.
To avoid the creation and the destruction of the internal data each time, we provide the class API :class:`~flory.mcmp.finder.CoexistingPhasesFinder`.
Using the class API usually involves three steps: creation of the finder instance, setting the system parameters and finding the coexisting states.
When the system sizes such as number of components :math:`N_\mathrm{C}` and the number of compartments :math:`N_\mathrm{M}` do not change, the finder can be reused.
Here we provide a simple example for generating a :math:`(\phi, \chi)` phase diagram for a simple binary mixture: 

.. literalinclude:: /../../examples/binary_phase_diagram.py
   :emphasize-lines: 13-22, 29-31
   :linenos:

We obtain the phase diagram

.. figure:: /../../examples/binary_phase_diagram.py.jpg
   :scale: 80 %
   :alt: phase diagram example

Check finite size effect
------------------------
The minimization process in our algorithm DO NOT guarantee that the equilibrium state is always found.
Due to the multistability of the multicomponent mixture, tt is possible that the algorithm find a local minimum.
For example, the true equilibrium state can be a 4-phase coexisting state, while the algorithm may find a metastable 3-phase coexisting state.
This issue can be resolved by launching more compartments than phases.
Obviously, the maximum number of phases can be found is always not larger than the number of the compartments.
By launching more compartments, it makes the algorithm much more likely to find the correct coexisting states.
Here we refer to this as the finite size effect, see an example of :math:`N_\mathrm{C}=8` below:

.. literalinclude:: /../../examples/finite_size_effect.py
   :emphasize-lines: 20-23
   :linenos:

We obtain

.. figure:: /../../examples/finite_size_effect.py.jpg
   :scale: 80 %
   :alt: phase diagram example

showing that :math:`N_\mathrm{M}=4` underestimates the number of phases in the final coexisting state, while larger :math:`N_\mathrm{M}` values give the correct result. 

Construct a ternary phase diagram
---------------------------------
Here we provide a simple example for generating a :math:`(\phi_B, \phi_N_\mathrm{A})` phase diagram for a simple ternary mixture with fixed interaction matrix.
The example first finds a point in the phase diagram that leads to three-phase coexistence, which is a triangle in the phase diagram.
Then starting from each edge of the triangle, we follow the direction to the unknown region in the phase diagram to complete all two-phase coexistence regions.

.. literalinclude:: /../../examples/ternary_phase_diagram.py
   :emphasize-lines: 11-20, 37-43, 59-62
   :linenos:

We obtain the phase diagram

.. figure:: /../../examples/ternary_phase_diagram.py.jpg
   :scale: 80 %
   :alt: phase diagram example

Using constraints
---------------------------------
In many systems such as mixtures containing ions or chemical reactions, there are additional constraints.
:mod:`flory` provides convenient ways to consider these constrains.
For example, in a system with 5 components, with first four components are charged, :mod:`flory` can find the coexisting phases that each phase is charge neutral, by applying a :class:`~flory.constraint.linear_local.LinearLocalConstraint`:

.. literalinclude:: /../../examples/coexisting_with_constraints.py
   :emphasize-lines: 17, 23
   :linenos:

Using different ensemble
---------------------------------
When considering an open system, the volume fractions are no longer conserved.
Instead, the system will keep fixed chemical potentials.
:mod:`flory` can handle this by switching from :class:`~flory.ensemble.canonical.CanonicalEnsemble` to :class:`~flory.ensemble.grandcanonical.GrandCanonicalEnsemble`:

.. literalinclude:: /../../examples/grandcanonical_with_constraints.py
   :emphasize-lines: 7, 13
   :linenos:
