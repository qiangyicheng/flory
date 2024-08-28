API reference
=============

The package :mod:`flory` is built around the core module :mod:`~flory.mcmp`, which
implements general finder :class:`~flory.mcmp.finder.CoexistingPhasesFinder` for
multiphase coexistence in multicomponent mixtures:

.. autosummary::
   :nosignatures:
   :template: custom-class-template.rst

   ~flory.mcmp

See :ref:`Implementation Details` for the implementation details of the finder
:class:`~flory.mcmp.finder.CoexistingPhasesFinder`.

The general finder :class:`~flory.mcmp.finder.CoexistingPhasesFinder` can be applied to a
combination of :mod:`~flory.interaction`, :mod:`~flory.entropy`, :mod:`~flory.ensemble`
and :mod:`~flory.constraint`: 

.. autosummary::
   :nosignatures:
   :template: custom-class-template.rst

   ~flory.interaction
   ~flory.entropy
   ~flory.ensemble
   ~flory.constraint

The module :mod:`~flory.free_energy` provides a comprehensive way to create interaction
and entropy instances, and provides more useful functionalities:

.. autosummary::
   :nosignatures:
   :template: custom-class-template.rst

   ~flory.free_energy


For usual systems, :mod:`~flory.shortcut` provide shortcuts for simple tasks of finding
coexisting phases:

.. autosummary::
   :nosignatures:
   :template: custom-class-template.rst

   ~flory.shortcut

.. rubric:: All Modules

Below, we list the full module structure of the package.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   flory