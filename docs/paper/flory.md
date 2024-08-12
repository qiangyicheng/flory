---
title: 'flory: A Python package for finding coexisting phases in multicomponent mixtures'
tags:
  - Python
  - physics
  - phase separation
  - free energy
authors:
  - name: Yicheng Qiang
    orcid: 0000-0003-2053-079X
    affiliation: 1
affiliations:
 - name: Max Planck Institute for Dynamics and Self-Organization, Göttingen, Germany
   index: 1
date: 11 August 2024
bibliography: flory.bib
---

# Summary

Phase separation is an intrinsic property of multicomponent systems and is widely observed in many fields, ranging from the simple demixing of oil and water to the condensation of biomolecules in cell.
In theory, the phase separation of multicomponent systems can lead to the coexistence of multiple phases, which in principle can be determined by solving the balancing equations between phases.
The information of coexisting phases can provide useful insights and act as the start point of more sophisticated models in many physical systems.  
However, due to the coupling the many nonlinear equations and the unknown number of coexisting phases in multicomponent mixtures, such task is nontrivial even with the minimal model, such as the Flory-Huggins model.

The `flory` python package presented in this paper provides convenient tools for the researchers to find the coexisting phases of multicomponent mixtures with Flory-Huggins interactions.
The package focuses on the coexisting phases in thermodynamically large systems, where the interfaces between phases can be safely ignored.
The package finds the coexisting phases by minimization the following average free energy density,

$$\bar{f}({N_\mathrm{p}}, \{J_\alpha\}, \{\phi_i^{(\alpha)}\}) = \sum_{\alpha=1}^{{N_\mathrm{p}}} J_\alpha f(\{\phi_i^{(\alpha)}\}) \; ,$$

where $N_\mathrm{c}$ is the number of components, $N_\mathrm{p}$ is the number of phases, $J_\alpha$ is the volume fraction of the phase $\alpha$ and $\phi_i^{(\alpha)}$ is the volume fraction of the component $i$ within the phase $\alpha$.
Each phase is considered to be homogeneous with a energy density

$$f(\{\phi_i\}) = \frac{1}{2}\sum_{i,j=1}^{N_\mathrm{c}} \chi_{ij} \phi_i \phi_j + \sum_{i=1}^{N_\mathrm{c}} \frac{\phi_i}{l_i} \ln \phi_i \; ,$$

where $\chi_{ij}$ is the Flory-Huggins interaction between component $i$ and $j$, and $l_i$ is the relative molecule size of the component $i$.
The package has the direct control over the average volume fractions of all components $\bar{\phi}_i = \sum_\alpha J_\alpha \phi_i^{(\alpha)}$, allows the researchers to find coexisting phases at a fixed system composition.

The main aim of `flory` package is to find the coexisting phases in multicomponent mixtures in a way that is efficient enough for constructing phase diagrams or sampling phase diagrams in high dimensions.
To reduce overhead, solver instances can be created and reused for multiple sets of parameters.
Besides, the core algorithms of the package are just-in-time compiled using numba [@lam2015Numba] to achieve high performance.
On top of this, function-style api is also provided for quick estimation.

# Methods

The main part of the `flory` package is the finder for coexisting phases that can be reused on different parameters once created, as soon as the system size is fixed, which is typical scenario while studying the phase behavior of mixtures.
For instance, the following code find the two coexisting phases of a symmetric binary mixture with $\chi=4$:
```python
finder = flory.CoexistingPhasesFinder(chis = [[0, 4.0], [4.0, 0]], phi_means = [0.5, 0.5], num_compartments = 16)
volumes, phis = finder.run()
```
where `num_compartments` is the main hyperparameter of the package, the number of compartments, which we will explain later.
Then the coexisting phases of another symmetric binary mixture with $\chi=3.5$ matrix can be obtained by:
```python
finder.chis = [[0, 3.5], [3.5, 0]]
volumes, phis = finder.run()
```
Such procedures can be repeated to complete certain tasks such as generating or sampling the phase diagrams efficiently.

The `flory` package applies to mixtures with arbitrary number of components.
The number of components is implied by the shape of the $\chi_{ij}$ matrix `chis` and the average volume fraction $\bar{\phi}_i$ `phi_means`. 
An optional parameter `sizes` is also provided to change the default relative molecule sizes $l_i$ in the mixture, which is useful to study the entropic effects of the components. 

The `flory` package is designed to deliver high performance.
All the key algorithms are just-in-time compiled using numba [@lam2015Numba].
Besides, the design of the solver separates most of the overheads to the creation of the solver, which can be amortized in many tasks.

The `flory` package adopts the latest numerical methods to determine the coexisting phases, based on the idea of compartments.
Compartments play the similar role of phases in calculation.
However, the phases need to have unique compositions while the compositions of the compartments do not have to [@zwicker2022Evolved].
By replacing unique phases by many more compartments (more than the number of phases allowed than the Gibbs phase rule), the package relieves the typical problem of multiple local minimum, and obtains the unique coexisting phases by clustering the coexisting compartments.
To reduce the influence of using many compartments on performance, the `flory` package implements the improved Gibbs ensemble method developed recently [@qiang2024Scaling].
This method redistributes components across all compartments in an improved way, such that the total computation cost only scales linearly with the number of compartments.
Therefore, the package can promisingly obtain the equilibrium coexisting states even though this cannot be guaranteed theoretically. 

Finally, `flory` package also provides a simple function-styled api to ease calculation of the coexisting phases in casual tasks:
```python
volumes, phis = flory.find_coexisting_phases([[0, 4.0], [4.0, 0]], [0.5, 0.5], 16) 
```
where the three parameters are the $\chi_{ij}$ matrix, the average volume fractions $\bar{\phi}_i$ and the number of compartments.

# References