Examples
======================

Here we provide a few examples for using package :mod:`flory`.

Find coexisting phases once
--------------------------------------------
Since we only need to find the coexisting phases once, we assign values to the :math:`\chi_{ij}` matrix and the average volume fractions :math:`\bar{\phi}_i` , and call the wrapper function :meth:`~flory.mcmp.find_coexisting_phases` directly.
Here we consider a symmetric two-component system, with :math:`\chi=4`:

.. literalinclude:: /../../examples/find_coexiting_phases.py
   :emphasize-lines: 6
   :linenos:

We obtain two symmetric phases:

.. literalinclude:: /../../examples/find_coexiting_phases.py.out
   :linenos:


By adding a :paramref:`~flory.mcmp.CoexistingPhasesFinder.sizes` parameter, we can also investigate the coexisting phases where the components have different molecule sizes:

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
--------------------------------------------
When constructing a phase diagram, we usually need to find coexisting phases for multiple instances.
To avoid the creation and the destruction of the internal data each time, we provide the class API :class:`~flory.mcmp.CoexistingPhasesFinder`.
Using the class API usually involves three steps: creation of the finder instance, setting the system parameters and finding the coexisting states.
When the system sizes such as number of components :math:`N_\mathrm{c}` and the number of compartments :math:`M` do not change, the finder can be reused.
Here we provide a simple example for generating a :math:`(\phi, \chi)` phase diagram for a simple binary mixture: 

.. literalinclude:: /../../examples/binary_phase_diagram.py
   :emphasize-lines: 11-13, 19,20
   :linenos:

We obtain the phase diagram

.. figure:: /../../examples/binary_phase_diagram.py.jpg
   :scale: 80 %
   :alt: phase diagram example

Check the finite size effect
--------------------------------------------
The minimization process in our algorithm DO NOT guarantee that the equilibrium state is always found.
Due to the multistability of the multicomponent mixture, tt is possible that the algorithm find a local minimum.
For example, the true equilibrium state can be a 4-phase coexisting state, while the algorithm may find a metastable 3-phase coexisting state.
This issue can be resolved by launching more compartments than phases.
Obviously, the maximum number of phases can be found is always not larger than the number of the compartments.
By launching more compartments, it makes the algorithm much more likely to find the correct coexisting states.
Here we refer to this as the finite size effect, see an example of :math:`N_\mathrm{c}=8` below:

.. literalinclude:: /../../examples/finite_size_effect.py
   :emphasize-lines: 19-22
   :linenos:

We obtain

.. figure:: /../../examples/finite_size_effect.py.jpg
   :scale: 80 %
   :alt: phase diagram example

showing that :math:`M=4` underestimates the number of phases in the final coexisting state, while larger :math:`M` values give the correct result. 

Construct a ternary phase diagram
--------------------------------------------
Here we provide a simple example for generating a :math:`(\phi_B, \phi_A)` phase diagram for a simple ternary mixture with fixed interaction matrix: 

.. literalinclude:: /../../examples/ternary_phase_diagram.py
   :emphasize-lines: 6, 40-60
   :linenos:

We obtain the phase diagram

.. figure:: /../../examples/ternary_phase_diagram.py.jpg
   :scale: 80 %
   :alt: phase diagram example
