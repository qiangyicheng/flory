# flory

[![Tests](https://github.com/qiangyicheng/flory/actions/workflows/python-package.yml/badge.svg)](https://github.com/qiangyicheng/flory/actions/workflows/python-package.yml)
[![Codecov](https://codecov.io/github/qiangyicheng/flory/graph/badge.svg?token=YF3K9ST8XQ)](https://codecov.io/github/qiangyicheng/flory)

`flory` is a Python package for determining multiple coexisting phases in multicomponent mixtures.
Currently `flory` supports finding coexisting phases of arbitrary multicomponent system with Flory-Huggins interactions in the canonical ensemble, whose average free energy density reads
$$
\bar{f}({N_\mathrm{p}}, \{J_\alpha\}, \{\phi_i^{(\alpha)}\}) = \sum_{\alpha=1}^{{N_\mathrm{p}}} J_\alpha f(\{\phi_i^{(\alpha)}\}) \; ,
$$
where $N_\mathrm{c}$ is the number of components, $N_\mathrm{p}$ is the number of phases, $J_\alpha$ is the volume fraction of the phase $\alpha$ and $ \phi_i^{(\alpha)}$ is the volume fraction of the component $i$ in the phase $\alpha$.
Each phase is considered as homogeneous, with its free energy density as
$$
f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{c}} \frac{\phi_i}{l_i} \ln \phi_i \; ,
$$
where $\chi_{ij}$ is the Flory-Huggins interaction between component $i$ and $j$, and $l_i$ is the relative molecule size of the component $i$.
With given interaction matrix $\chi_{ij}$, module `flory.mcmp` simply finds the coexisting states of a given point in the phase diagram, namely a given average volume fractions of all components across the system $\bar{\phi}_i$.

Installation
------------

`flory` is available on `pypi`, so you should be able to install it through `pip`:

```bash
pip install flory
```
As an alternative, you can install `flory` through [conda](https://docs.conda.io/en/latest/)
using the [conda-forge](https://conda-forge.org/) channel:

```bash
conda install -c conda-forge flory
```

Installation with `conda` includes all dependencies of `flory`.

Usage
-----

A simple example finds the coexisting phases of a binary mixture:

```python
import flory

chis = [[0, 4.0], [4.0, 0]]     # Set the \chi matrix
phi_means = [0.5, 0.5]          # Set the average volume fractions

volumes, phis = flory.find_coexisting_phases(chis, phi_means, 16)   # obtain the relative volumes and the compositions of the two coexisting phases
```


More information
----------------
* See examples in [examples folder](https://github.com/qiangyicheng/flory/tree/main/examples)


