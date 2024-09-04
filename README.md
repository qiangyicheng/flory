# flory

[![Tests](https://github.com/qiangyicheng/flory/actions/workflows/python-package.yml/badge.svg)](https://github.com/qiangyicheng/flory/actions/workflows/python-package.yml)
[![Codecov](https://codecov.io/github/qiangyicheng/flory/graph/badge.svg?token=YF3K9ST8XQ)](https://codecov.io/github/qiangyicheng/flory)
[![Documentation Status](https://readthedocs.org/projects/flory/badge/?version=latest)](https://flory.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/flory.svg)](https://badge.fury.io/py/flory)

`flory` is a Python package for analyzing field theories of multicomponent mixtures.
In particular, the package provides routines to determine coexisting states numerically, which is a challenging problem since the thermodynamic coexistence conditions are many coupled non-linear equations.
`flory` supports finding coexisting phases with an arbitrary number of components.
The associated average free energy density of the system reads

$$\bar{f}({N_\mathrm{P}}, \{J_p\}, \{\phi_{p,i}\}) = \sum_{p=1}^{{N_\mathrm{P}}} J_p f(\{\phi_{p,i}\}) \; ,$$

where $N_\mathrm{C}$ is the number of components, $N_\mathrm{P}$ is the number of phases, $J_p$ is the fraction of the system volume occupied by phase $p$, and $\phi_{p,i}$ is the volume fraction of component $i$ in phase $p$.

`flory` supports different forms of interaction, entropy, ensemble, and constraints to describe the free energy of phases.
For example, with the commonly used Flory-Huggins free energy, the free energy density of each homogeneous phase reads

$$f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{C}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{C}} \frac{\phi_i}{l_i} \ln \phi_i \; ,$$

where $\chi_{ij}$ is the Flory-Huggins interaction parameter between component $i$ and $j$, and $l_i$ is the relative molecular size of component $i$.
Given an interaction matrix $\chi_{ij}$, average volume fractions of all components across the system $\bar{\phi}_i$, and the relative molecule sizes $l_i$, `flory` provides tools to find the coexisting phases in equilibrium.

Installation
------------
`flory` is available on `pypi`, so you should be able to install it through `pip`:

```bash
pip install flory
```

Usage
-----
The following example determines the coexisting phases of a binary mixture with Flory-Huggins free energy:

```python
import flory

num_comp = 2                    # Set number of components
chis = [[0, 4.0], [4.0, 0]]     # Set the \chi matrix
phi_means = [0.5, 0.5]          # Set the average volume fractions

# obtain coexisting phases
phases = flory.find_coexisting_phases(num_comp, chis, phi_means)
```

It is equivalent to a more advanced example:

```python
import flory

num_comp = 2                    # Set number of components
chis = [[0, 4.0], [4.0, 0]]     # Set the \chi matrix
phi_means = [0.5, 0.5]          # Set the average volume fractions

# create a free energy
fh = flory.FloryHuggins(num_comp, chis)
# create a ensemble
ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
# construct a finder from interaction, entropy and ensemble
finder = flory.CoexistingPhasesFinder(fh.interaction, fh.entropy, ensemble)
# obtain phases by clustering compartments 
phases = finder.run().get_clusters()
```

The free energy instance provides more tools for analysis, such as:
```python
# calculate the chemical potentials of the coexisting phases
mus = fh.chemical_potentials(phases.fractions)
```

More information
----------------
* See examples in [examples folder](https://github.com/qiangyicheng/flory/tree/main/examples)
* [Full documentation on readthedocs](https://flory.readthedocs.io/)
