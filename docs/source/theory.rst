Theory
======================

Finding coexisting phases is a long lasting topic in many fields, such as `calculating
phase diagrams <https://doi.org/10.1017/CBO9780511804137>`_ for alloys. Theoretically, the
coexisting phases can be found by minimizing the mean free energy density of the whole
mixture. In a perfect world, an ideal method for finding coexisting phase would

- allow arbitrary number of components :math:`N_\mathrm{C}`;
- allow each phase to have an arbitrary free energy;
- always locate the global minimum.

Although such a method may exist theoretically, it would not be numerically feasible due
to the high-dimensional nature of the phase space of multicomponent mixtures with many
components. Indeed, finding coexisting phases are usually `limited to mixtures of few
components <https://doi.org/10.1021/acs.jctc.3c00198>`_, while recent researches,
especially those in biophysics and soft matter physics, usually involves phase separation
of mixtures with many components. Package :mod:`flory` is designed exactly for this
scenario. Instead of trying to cover arbitrary multicomponent mixtures, package
:mod:`flory` restricts its application to the mixtures

- with uniform free energy function;
- using volume fractions as order parameters;
- containing entropic energy similar to ideal gas.

These requirements are satisfied in many theoretical studies. By assuming the restriction
above, package :mod:`flory` implements an efficient method that is suitable for mixtures
with arbitrary interaction, ensemble, constraint, and a limit set of entropic energy. The
method finds the most favorable coexisting phases by redistributing all components around
all phases at the same time, guided by the physical information. To avoid searching for
the entire free energy surface, the method chooses not to guarantee that the global
minimum is always found, but aims to find it in high possibility.

Concepts
---------------------

For a general mixture, the equilibrium coexisting states can be obtained by iteratively
optimizing the average free energy 

.. math::
   \bar{f}({N_\mathrm{P}}, \{J_p\}, \{\phi_{p,i}\}) = \sum_{p=1}^{{N_\mathrm{P}}} J_p f(\{\phi_{p,i}\}) \; ,

where :math:`N_\mathrm{P}` is the number of phases, :math:`J_p` is the volume of phase
:math:`p` and :math:`\phi_{p,i}` is the volume fraction of the component :math:`i` in
phase :math:`p`. To have an efficient method, package :mod:`flory` using following
concepts:

- Free energy :math:`f` consists of four parts: **interaction**, **entropy**,
  **ensemble** and **constraint**.
- Instead of optimizing the number of phases :math:`N_\mathrm{P}`, package fixes the number
  of **compartments** :math:`N_\mathrm{M}`. Compartments can be viewed to be equivalence of phases
  but do not have to be unique. To obtain the correct coexisting phases, the package
  usually uses number of compartments :math:`N_\mathrm{M}` much larger than the maximum number of
  phases :math:`N_\mathrm{P}` allowed by the Gibbs phase law.
- :math:`N_\mathrm{C}` **components** interact and feel the constraints though the volume
  fractions of :math:`N_\mathrm{S}` **features**. This may reduce computational cost
  significantly for many systems. 

Inspired by `polymeric field theories <https://doi.org/10.1088/0953-8984/10/37/002>`_, we
express the free energy of the entire mixtures in an extended form by introducing the
conjugate variables :math:`w_r^{(m)}`,

.. math::
    &\bar{f}(\{J_m\}, \{\phi_r^{(m)}\}, \{w_r^{(m)}\}, \xi) \\
    =& 
    \sum_{m=1}^{N_\mathrm{M}} J_m \biggl[ 
        f_\mathrm{interaction}(\{\phi_r^{(m)}\}) 
        - \sum_r^{N_\mathrm{S}} \phi_r^{(m)} w_r^{(m)} 
        + \xi\biggl(\sum_r^{N_\mathrm{S}} \phi_r^{(m)} -1\biggr) 
        \biggr] \\
        &+ g\left(\{Q_i(\{w_r^{(m)}\})\}\right) 
        + f_\mathrm{constraint}(\{\phi_r^{(m)}\})
        + \eta\biggl(\sum_{m=1}^{N_\mathrm{M}} J_m -1\biggr) \; .

Here,

- :math:`f_\mathrm{interaction}(\{\phi_r^{(m)}\})` describes the **interaction** energy
  density in each compartment.
- :math:`- \sum_r^{N_\mathrm{S}} \phi_r^{(m)} w_r^{(m)}` couples the volume fractions of
  features :math:`\phi_r^{(m)}` to their conjugate variable :math:`w_r^{(m)}`.
- :math:`\xi(\sum_r^{N_\mathrm{S}} \phi_r^{(m)} -1)` enforces the incompressibility
  through the Lagrange multiplier :math:`\xi`
- :math:`Q_i(\{w_r^{(m)}\})` is the single molecule partition function of the component
  :math:`i` under the mean field :math:`w_r^{(m)}`, describing the translational
  **entropy** of the component :math:`i`.
- :math:`g(\{Q_i(\{w_r^{(m)}\})\})` maps the single molecule partition
  functions to part of the entropic energies, whose form is defined by the **ensemble**.
- :math:`f_\mathrm{constraint}(\{\phi_r^{(m)}\})` describes additional **constraints** that
  the system will be subject to.
- :math:`\eta(\sum_{m=1}^{N_\mathrm{M}} J_m -1)` conserves the total volume.

Although the form above may look different from the normal free energy function, we next
use an example of Flory-Huggins free energy to show their equivalence and the iteration scheme
it leads to.

Example: Flory-Huggins Free Energy
-------------------------------------

The Flory-Huggins free energy of a single compartment reads

.. math::
   f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{C}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{C}} \frac{\phi_i}{l_i} \ln \phi_i \; .

Here we assume that the components all interact through their own features. In other
words, the number of components is the same as the number of features. In the canonical
ensemble, the average volume fractions of components across all compartments are kept
constant. Therefore, we express the mean free energy into,

.. math::
    \bar{f} = & \sum_{m=1}^{N_\mathrm{M}} J_m \biggl[\frac{1}{2} \sum_{i,j=1}^{N_\mathrm{C}} \chi_{ij} \phi_i^{(m)} \phi_j^{(m)} - \sum_{i=1}^{N_\mathrm{C}} w_i^{(m)}\phi_i^{(m)} + \xi_m \biggl(\sum_{i=1}^{N_\mathrm{C}} \phi_i^{(m)}-1\biggr) \biggr] \\
    &- \sum_{i=1}^{N}\bar{\phi_i}\ln Q_i +\eta\biggl(\sum_{m=1}^{N_\mathrm{M}} J_m -1\biggr) \;,

with the single molecule partition function

.. math::
    Q_i = \sum_{m=1}^{N_\mathrm{M}} J_m \exp\left(-w_i^{(m)}\right).

Here we recall that :math:`J_m` are the relative volumes of the compartments,
:math:`w_i^{(m)}` are the conjugate variables of :math:`\phi_i^{(m)}`, and :math:`\xi_m`
and :math:`\eta` are the Lagrangian multipliers for incompressibility of each compartment
and compartment volume conservation, respectively. Consequently, the extremum of the
extended free energy with respect to :math:`\xi(x)` corresponds to incompressibility,

.. math::
    \frac{\partial \bar{f}}{\partial \xi_m} \propto \biggl(\sum_{i=1}^{N_\mathrm{C}} \phi_i^{(m)} - 1\biggr)J_m = 0 \quad \quad \Rightarrow  \quad \quad \sum_{i=1}^{N_\mathrm{C}} \phi_i^{(m)} = 1 \;,

the extremum with respect to :math:`\eta` corresponds to conservation of the total volume
of all compartments,

.. math::
    \frac{\partial \bar{f}}{\partial \eta} \propto \sum_{m=1}^{N_\mathrm{M}} J_m -1 = 0 \quad \quad \Rightarrow  \quad \quad \sum_{m=1}^{N_\mathrm{M}} J_m = 1 \;,

and the extremum with respect to :math:`w_i^{(m)}` defines the relationship between
:math:`\phi_i^{(m)}` and :math:`w_i^{(m)}`,

.. math::
    \frac{\partial \bar{f}}{\partial w_i^{(m)}} \propto -\phi_i^{(m)} J_m + \frac{\bar{\phi_i}}{Q_i}\exp\left(-w_i^{(m)} \right) J_m = 0  \quad \quad \Rightarrow  \quad \quad  \phi_i^{(m)} = \frac{\bar{\phi_i}}{Q_i}\exp\left(-w_i^{(m)}\right) .

By inserting three equations above into the extended free energy density, the original
free energy density is recovered except for a constant, which has no influences on
thermodynamics. Therefore, minimizing the extended free energy function will naturally
lead to balanced chemical potentials and osmotic pressures, and it is unnecessary to
consider them explicitly. To optimize the extended free energy density, we obtain the
self-consistent equations

.. math::
    1            & = \sum_{i=1}^{N_\mathrm{C}} \phi_i^{(m)}                                       \\
    1            & = \sum_{m=1}^{N_\mathrm{M}} J_m                                                   \\
    \phi_i^{(m)} & = \frac{\bar{\phi_i}}{Q_i}\exp\left(-w_i^{(m)}\right)              \\
    w_i^{(m)}    & = \sum_{j=1}^{N_\mathrm{C}} \chi_{ij} \phi_j^{(m)} + \xi_m                     \\
    -\eta        & = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{C}} \chi_{ij} \phi_i^{(m)} \phi_j^{(m)}
    - \sum_{i=1}^{N_\mathrm{C}} w_i^{(m)}\phi_i^{(m)}
    + \xi_m \biggl(\sum_{i=1}^{N_\mathrm{C}} \phi_i^{(m)}-1\biggr)
    - \sum_{i=1} \phi_i^{(m)}\; .

To solve these equations, we design the following iterative scheme

.. math::
    Q_i^{(m)}    & = \sum_{m=1}^{N_\mathrm{M}} \exp\left(-w_i^{(m)}\right) J_m                                                            \\
    \phi_i^{(m)} & = \frac{\bar{\phi_i}}{Q_i^{(m)}}\exp\left(-w_i^{(m)}\right)                                             \\
    \xi_m        & = \frac{1}{{N_\mathrm{C}}} \biggl(\sum_{i=1}^{{N_\mathrm{C}}} w_i^{(m)} - \sum_{i,j=1}^{{N_\mathrm{C}}} \chi_{ij} \phi_j^{(m)} \biggr) \\
    \eta_m       & = -\frac{1}{2}\sum_{i,j=1}^{N_\mathrm{C}} \chi_{ij} \phi_i^{(m)} \phi_j^{(m)}
    + \sum_{i=1}^{N_\mathrm{C}} w_i^{(m)}\phi_i^{(m)} - \xi_m \biggl(\sum_{i=1}^{N_\mathrm{C}} \phi_i^{(m)}-1\biggr)
    + \sum_{i=1} \phi_i^{(m)}                                                                                                  \\
    \bar{\eta}   & = \sum_{m=1}^{N_\mathrm{M}}  \eta_m J_m                                                                                \\
    w_i^{(m)*}   & = \sum_{j=1}^{N_\mathrm{C}} \chi_{ij} \phi_j^{(m)} + \xi^{(m)}                                                      \\
    J_m^*        & = J^{(m)} + \eta^{(m)} - \bar{\eta}\;,

where the asterisks denote the output of the iteration. In order to improve numerical
stability, we also adopt the simple mixing strategy,

.. math::
    w_i^{(m),\mathrm{new}} & = w_i^{(m)} + p \left(w_i^{(m)*} - w_i^{(m)}\right) \\
    J^{(m),\mathrm{new}}   & = J_m + \beta \left(J_m^* - J_m\right)\;,

where :math:`p` and :math:`\beta` are two empirical constants, which are termed
:paramref:`~flory.mcmp.finder.CoexistingPhasesFinder.acceptance_omega` and
:paramref:`~flory.mcmp.finder.CoexistingPhasesFinder.acceptance_Js` and usually chosen near
:math:`10^{-3}`. We note again that in such iteration scheme the problem of negative
volume fractions is relieved. However, there is no guarantee that relative compartment
volume :math:`J_m` is always positive. Although the method does not suffer from
negative :math:`J_m`, negative :math:`J_m` implies that the system might be outside of the
allowed region on the tie hyperplane. To alleviate this, we always use :math:`\beta`
smaller than :math:`p`, and adopt a killing-and-revive strategy to correct the worst
cases: Once :math:`J_m` is found to be negative at certain :math:`m`, e.g. :math:`m_0`,
the corresponding compartment is considered "dead" and is going to be revived by resetting
:math:`J_{m_0}` to its initial value, and the corresponding :math:`w_i^{(m_0)}` will be
redrawn from random distributions. To obey volume conservation, all other :math:`J_m` will
be renormalized. The same scheme is used to initialize the simulation, i.e., all
compartments are considered "dead" at the beginning of the simulation.

As we mentioned, this method does not guarantee that the true equilibrium state (the
global minimum) is always found. Therefore, :mod:`flory` handles the problem by launching
many more compartments than the number of components, :math:`N_\mathrm{M}\gg{N_\mathrm{C}}`, see
:paramref:`~flory.mcmp.finder.CoexistingPhasesFinder.num_part`.