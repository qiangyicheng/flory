Theory
======================
There are multiple challenges for determining coexisting states in equilibrium in multi-component mixture, even the free energy of each phase has a analytical form.
First, the optimization problem is high-dimensional.
Second, different types of constrains need to be satisfied, including the incompressibility and volume conservation.
In addition, it is generally difficult to conclude whether the obtained coexisting states are the true equilibrium or metastable states due to multistability.

`Previously <https://doi.org/10.1073/pnas.2201250119>`_, the coexisting states are usually obtained by finding the balance of chemical potentials and osmotic pressures between all pairs of phases.
Instead of solving the balance equations of chemical potentials and osmotic pressures, :mod:`mcmp` adopts the more fundamental idea of free energy minimization, which is the origin of the balance equations.
The equilibrium coexisting states can be obtained by optimizing the average free energy 

.. math::
   \bar{f}({N_\mathrm{p}}, \{J_\alpha\}, \{\phi_i^{(\alpha)}\}) = \sum_{\alpha=1}^{{N_\mathrm{p}}} J_\alpha f(\{\phi_i^{(\alpha)}\}) \; ,

.. math::
   f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{c}} \frac{\phi_i}{l_i} \ln \phi_i \; ,

over all possible phase counts :math:`N_\mathrm{p}`, phase volume fractions :math:`J_\alpha` and phase compositions :math:`\phi_i^{(\alpha)}`.
To allow variable phase counts, we consider an ensemble of :math:`M` abstract compartments, where :math:`M` is much larger than the number of components :math:`N_\mathrm{p}`.
Inspired by `polymeric field theories <https://doi.org/10.1088/0953-8984/10/37/002>`_, :mod:`mcmp` alleviates the problem of negative volume fractions during the relaxation dynamics and conserves the average volume fractions by extending the free energy into

.. math::
    \bar{f} = 
    & \sum_{m=1}^M J_m \left[\frac{1}{2} \sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i^{(m)} \phi_j^{(m)} - \sum_{i=1}^{N_\mathrm{c}} w_i^{(m)}\phi_i^{(m)} + \xi_m \biggl(\sum_{i=1}^{N_\mathrm{c}} \phi_i^{(m)}-1\biggr) \right] \\
    &- \sum_{i=1}^{N}\bar{\phi_i}\ln Q_i +\eta\biggl(\sum_{m=1}^M J_m -1\biggr) \;,

with the single molecule partition function

.. math::
    Q_i = \sum_{m=1}^M J_m \exp\left(-w_i^{(m)}\right).

Here, :math:`J_m` are the relative volumes of the compartments, :math:`w_i^{(m)}` are the conjugate variables of :math:`\phi_i^{(m)}`, and :math:`\xi_m` and :math:`\eta` are the Lagrangian multipliers for incompressibility of each compartment and compartment volume conservation, respectively.
Consequently, the extremum of the extended free energy with respect to :math:`\xi(x)` corresponds to incompressibility,

.. math::
    \frac{\partial \bar{f}}{\partial \xi_m} \propto \biggl(\sum_{i=1}^{N_\mathrm{c}} \phi_i^{(m)} - 1\biggr)J_m = 0 \quad \quad \Rightarrow  \quad \quad \sum_{i=1}^{N_\mathrm{c}} \phi_i^{(m)} = 1 \;,

the extremum with respect to :math:`\eta` corresponds to conservation of the total volume of all compartments,

.. math::
    \frac{\partial \bar{f}}{\partial \eta} \propto \sum_{m=1}^M J_m -1 = 0 \quad \quad \Rightarrow  \quad \quad \sum_{m=1}^M J_m = 1 \;,

and the extremum with respect to :math:`w_i^{(m)}` defines the relationship between :math:`\phi_i^{(m)}` and :math:`w_i^{(m)}`,

.. math::
    \frac{\partial \bar{f}}{\partial w_i^{(m)}} \propto -\phi_i^{(m)} J_m + \frac{\bar{\phi_i}}{Q_i}\exp\left(-w_i^{(m)} \right) J_m = 0  \quad \quad \Rightarrow  \quad \quad  \phi_i^{(m)} = \frac{\bar{\phi_i}}{Q_i}\exp\left(-w_i^{(m)}\right) .

By inserting three equations above into the extended free energy density, the original free energy density is recovered except for a constant, which has no influences on thermodynamics.
Therefore, minimizing the extended free energy function will naturally lead to balanced chemical potentials and osmotic pressures, and it is unnecessary to consider them explicitly.
To optimize the extended free energy density, we obtain the self-consistent equations

.. math::
    1            & = \sum_{i=1}^{N_\mathrm{c}} \phi_i^{(m)}                                       \\
    1            & = \sum_{m=1}^M J_m                                                   \\
    \phi_i^{(m)} & = \frac{\bar{\phi_i}}{Q_i}\exp\left(-w_i^{(m)}\right)              \\
    w_i^{(m)}    & = \sum_{j=1}^{N_\mathrm{c}} \chi_{ij} \phi_j^{(m)} + \xi_m                     \\
    -\eta        & = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i^{(m)} \phi_j^{(m)}
    - \sum_{i=1}^{N_\mathrm{c}} w_i^{(m)}\phi_i^{(m)}
    + \xi_m \biggl(\sum_{i=1}^{N_\mathrm{c}} \phi_i^{(m)}-1\biggr)
    - \sum_{i=1} \phi_i^{(m)}\; .

To solve these equations, we design the following iterative scheme

.. math::
    Q_i^{(m)}    & = \sum_{m=1}^M \exp\left(-w_i^{(m)}\right) J_m                                                            \\
    \phi_i^{(m)} & = \frac{\bar{\phi_i}}{Q_i^{(m)}}\exp\left(-w_i^{(m)}\right)                                             \\
    \xi_m        & = \frac{1}{{N_\mathrm{c}}} \left(\sum_{i=1}^{{N_\mathrm{c}}} w_i^{(m)} - \sum_{i,j=1}^{{N_\mathrm{c}}} \chi_{ij} \phi_j^{(m)} \right) \\
    \eta_m       & = -\frac{1}{2}\sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i^{(m)} \phi_j^{(m)}
    + \sum_{i=1}^{N_\mathrm{c}} w_i^{(m)}\phi_i^{(m)} - \xi_m \biggl(\sum_{i=1}^{N_\mathrm{c}} \phi_i^{(m)}-1\biggr)
    + \sum_{i=1} \phi_i^{(m)}                                                                                                  \\
    \bar{\eta}   & = \sum_{m=1}^M  \eta_m J_m                                                                                \\
    w_i^{(m)*}   & = \sum_{j=1}^{N_\mathrm{c}} \chi_{ij} \phi_j^{(m)} + \xi^{(m)}                                                      \\
    J_m^*        & = J^{(m)} + \eta^{(m)} - \bar{\eta}\;,

where the asterisks denote the output of the iteration.
In order to improve numerical stability, we also adopt the simple mixing strategy,

.. math::
    w_i^{(m),\mathrm{new}} & = w_i^{(m)} + \alpha \left(w_i^{(m)*} - w_i^{(m)}\right) \\
    J^{(m),\mathrm{new}}   & = J_m + \beta \left(J_m^* - J_m\right)\;,

where :math:`\alpha` and :math:`\beta` are two empirical constants, which are termed :paramref:`~flory.mcmp.CoexistingPhasesFinder.acceptance_omega` and :paramref:`~flory.mcmp.CoexistingPhasesFinder.acceptance_Js` and usually chosen near :math:`10^{-3}`.
We note again that in such iteration scheme the problem of negative volume fractions is relieved.
However, there is no guarantee that relative compartment volume :math:`J_m` is always positive.
Although the algorithm does not suffer from negative :math:`J_m`, negative :math:`J_m` implies that the system might be outside of the allowed region on the tie hyperplane.
To alleviate this, we always use :math:`\beta` smaller than :math:`\alpha`, and adopt a killing-and-revive strategy to correct the worst cases:
Once :math:`J_m` is found to be negative at certain :math:`m`, e.g. :math:`m_0`, the corresponding compartment is considered "dead" and is going to be revived by resetting :math:`J_{m_0}` to its initial value, and the corresponding :math:`w_i^{(m_0)}` will be redrawn from random distributions.
To obey volume conservation, all other :math:`J_m` will be renormalized.
The same scheme is used to initialize the simulation, i.e., all compartments are considered "dead" at the beginning of the simulation.

Due to multistability, this algorithm does not guarantee that the true equilibrium state is always found.
Therefore :mod:`mcmp` handles the problem of multistability by launching many more compartments than the number of components, :math:`M\gg{N_\mathrm{c}}`, see :paramref:`~flory.mcmp.CoexistingPhasesFinder.num_compartments`.