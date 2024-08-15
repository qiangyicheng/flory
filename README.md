# flory

[![Tests](https://github.com/qiangyicheng/flory/actions/workflows/python-package.yml/badge.svg)](https://github.com/qiangyicheng/flory/actions/workflows/python-package.yml)
[![Codecov](https://codecov.io/github/qiangyicheng/flory/graph/badge.svg?token=YF3K9ST8XQ)](https://codecov.io/github/qiangyicheng/flory)
[![Documentation Status](https://readthedocs.org/projects/flory/badge/?version=latest)](https://flory.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/flory.svg)](https://badge.fury.io/py/flory)

`flory` is a Python package for analyzing theories of multicomponent mixtures.
In particular, the package provides routines to determine coexisting states numerically.
This is a challenging problem, since the thermodynamic coexistence conditions are many coupled non-linear equations.
Currently, `flory` supports finding coexisting phases in the canonical ensemble of system with an arbitrary number of components that interact according to Flory-Huggins theory.
The associated average free energy density of the system reads

$$\bar{f}({N_\mathrm{p}}, \{J_\alpha\}, \{\phi_i^{(\alpha)}\}) = \sum_{\alpha=1}^{{N_\mathrm{p}}} J_\alpha f(\{\phi_i^{(\alpha)}\}) \; ,$$

where $N_\mathrm{c}$ is the number of components, $N_\mathrm{p}$ is the number of phases, $J_\alpha$ is the fraction of the system volume occupied by phase $\alpha$, and $\phi_i^{(\alpha)}$ is the volume fraction of component $i$ in phase $\alpha$.
The free energy density of each homogeneous phase reads

$$f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{c}} \frac{\phi_i}{l_i} \ln \phi_i \; ,$$

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
A simple example determines the coexisting phases of a binary mixture:

```python
import flory

chis = [[0, 4.0], [4.0, 0]]     # Set the \chi matrix
phi_means = [0.5, 0.5]          # Set the average volume fractions

# obtain relative volumes and compositions of the two coexisting phases
volumes, phis = flory.find_coexisting_phases(chis, phi_means, 16)
```

More information
----------------
* See examples in [examples folder](https://github.com/qiangyicheng/flory/tree/main/examples)
* [Full documentation on readthedocs](https://flory.readthedocs.io/)
