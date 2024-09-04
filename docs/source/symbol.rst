Symbols and Conventions
========================

Here we collect the physical meaning of the symbols and the corresponding storage
convention use in package :mod:`flory`. Note that the first column is always dimensionless.

Counts
---------------------------

.. list-table:: Symbols, Physical Meaning and Conventions: Counts
    :widths: 15 15 40 30
    :header-rows: 1

    * - Symbol
      - Unit
      - Physical Meaning
      - Convention
    * - :math:`N_\mathrm{C}`
      - 1
      - Number of components
      - Integer
    * - :math:`N_\mathrm{S}`
      - 1
      - Number of features
      - Integer
    * - :math:`N_\mathrm{P}`
      - 1
      - Number of phases
      - Integer
    * - :math:`N_\mathrm{U}`
      - 1
      - Number of unstable modes
      - Integer
    * - :math:`N_\mathrm{A}`
      - 1
      - Number of constraints
      - Integer
    * - :math:`N_\mathrm{M}`
      - 1
      - Number of compartments
      - Integer
    * - :math:`i,j,k`
      - 1
      - Indexes for components. 1-Based in documentation
      - Integer :math:`\in [1, N_\mathrm{C}]`
    * - :math:`r,s,t`
      - 1
      - Indexes for features. 1-Based in documentation
      - Integer :math:`\in [1, N_\mathrm{S}]`
    * - :math:`p`
      - 1
      - Index for phases. 1-Based in documentation
      - Integer :math:`\in [1, N_\mathrm{P}]`
    * - :math:`\alpha`
      - 1
      - Index for constraints. 1-Based in documentation
      - Integer :math:`\in [1, N_\mathrm{A}]`
    * - :math:`m`
      - 1
      - Index for compartments. 1-Based in documentation
      - Integer :math:`\in [1, N_\mathrm{M}]`

Common Variables
---------------------------
Here we collect the common symbols used in the user interface.

.. list-table:: Symbols, Physical Meaning and Conventions: Common Variables
    :widths: 15 15 40 30
    :header-rows: 1

    * - Symbol
      - Unit
      - Physical Meaning
      - Convention
    * - :math:`\phi_i`
      - 1
      - Volume fraction of component :math:`i`.
      - :math:`N_\mathrm{C}`
    * - :math:`\mu_i`
      - :math:`k_\mathrm{B}T/\nu`
      - Chemical potential of component :math:`i` by unit volume.
      - :math:`N_\mathrm{C}`
    * - :math:`\phi_{p,i}`
      - 1
      - Volume fraction of component :math:`i` in compartment :math:`p`.
      - :math:`N_\mathrm{P} \times N_\mathrm{C}`
    * - :math:`\bar{\phi}_i`
      - 1
      - Average volume fraction of component :math:`i`.
      - :math:`N_\mathrm{C}`
    * - :math:`l_i`
      - :math:`\nu`
      - Relative molecule volume of component :math:`i`.
      - :math:`N_\mathrm{C}`
    * - :math:`J_p`
      - Total :math:`V`
      - Relative volume of phase :math:`p`.
      - :math:`N_\mathrm{P}`
    * - :math:`f`
      - :math:`k_\mathrm{B}T/\nu`
      - Free energy density.
      - :math:`N_\mathrm{P}`

Advanced Variables
---------------------------
Here we collect the advances symbols used in the package. Note that the major difference
from the common ones is that in most of cases, the index of the compartment :math:`m` is
the last index, namely the fastest-varying index (C-convention), to improve performance.

.. list-table:: Symbols, Physical Meaning and Conventions: Advanced Variables
    :widths: 15 15 40 30
    :header-rows: 1

    * - Symbol
      - Unit
      - Physical Meaning
      - Convention
    * - :math:`\phi_r`
      - 1
      - Volume fraction of feature :math:`r`.
      - :math:`N_\mathrm{S}`
    * - :math:`\bar{\phi}_r`
      - 1
      - Mean volume fraction of feature :math:`r`.
      - :math:`N_\mathrm{S}`
    * - :math:`\phi_i^{(m)}`
      - 1
      - Volume fractions of component :math:`i` in compartment :math:`m`.
      - :math:`N_\mathrm{C} \times N_\mathrm{M}`
    * - :math:`\phi_r^{(m)}`
      - 1
      - Volume fractions of feature :math:`r` in compartment :math:`m`.
      - :math:`N_\mathrm{S} \times N_\mathrm{M}`
    * - :math:`w_r^{(m)}`
      - 1
      - Conjugate variable of (mean field felt by) :math:`\phi_r^{(m)}`.
      - :math:`N_\mathrm{S} \times N_\mathrm{M}`
    * - :math:`p_i^{(m)}`
      - 1
      - Boltzmann factor of component :math:`i` in compartment :math:`m`.
      - :math:`N_\mathrm{C} \times N_\mathrm{M}`
    * - :math:`Q_i`
      - 1
      - Single molecule partition function of component :math:`i`.
      - :math:`N_\mathrm{C}`
    * - :math:`J_m`
      - Arbitrary volume
      - Relative volume of compartment :math:`m`.
      - :math:`N_\mathrm{M}`
    * - :math:`C_{\alpha,r}`
      - 1
      - Coefficients of features for linear constraint :math:`\alpha`.
      - :math:`N_\mathrm{A} \times N_\mathrm{S}`
    * - :math:`T_\alpha`
      - 1
      - Target (right-hand-side) of constraint :math:`\alpha`.
      - :math:`N_\mathrm{A}`
    * - :math:`\kappa`
      - 1
      - Elasticity of constraints.
      - Scaler
