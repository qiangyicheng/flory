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
  - name: David Zwicker
    orcid: 0000-0002-3909-3334
    affiliation: 1
affiliations:
 - name: Max Planck Institute for Dynamics and Self-Organization, Göttingen, Germany
   index: 1
date: 9 September 2024
bibliography: paper.bib
---

# Summary

Phase separation, an intrinsic property of mixtures, is widely observed in many scenarios, ranging from the simple demixing of oil and water to the condensation of biomolecules in cells.
In multicomponent mixtures, phase separation leads to the coexistence of multiple phases.
Finding these coexisting phases is a general question in many fields such as chemical engineering [@lukas2007Computational] and soft matter physics [@jacobs2023Theory].
There are several strategies to theoretically predict the coexisting phases [@jacobs2023Theory], including direct spatially-resolved simulations [@shrinivas2021Phase], the construction of the convex hull of the free energy  [@mao2019Phase], solving the balance equations [@zwicker2022Evolved], and direct minimization of the free energy [@lukas2007Computational].
However, the computational cost of existing methods scales strongly with component count, so systems with many different components cannot be analyzed.
The `flory` package provides an easily accessible, performant, and extensible code that allows researchers to address such problem flexibly.

The `flory` Python package provides convenient tools for finding coexisting phases of a broad range of multicomponent mixtures.
The package focuses on coexisting phases in thermodynamically large systems, where the interfaces between phases become negligible.
The package finds coexisting phases by minimizing the average free energy density $\bar f$ of the entire system, given by

$$\bar{f}({N_\mathrm{P}}, \{J_p\}, \{\phi_{p,i}\}) = \sum_{p=1}^{{N_\mathrm{P}}} J_p f(\{\phi_{p,i}\}) \; ,$$

where $N_\mathrm{C}$ is the number of components, $N_\mathrm{P}$ is the number of phases, $J_p$ denotes the fraction of volume that phase $p=1,\ldots,N_\mathrm{P}$ occupies in the entire system, and $\phi_{p,i}$ is the volume fraction of component $i=1,\ldots,N_\mathrm{C}$ in phase $p$.
The physical behavior of the mixture is encoded in the free energy density $f$, which `flory` expresses using four orthogonal aspects: interaction, entropy, ensemble, and constraints.
The package only imposes limits on the entropy part, which is crucial for the core algorithm, while the other three aspects are rather flexible.
By combining these four aspects, `flory` supports a broad range of free energy densities $f$ with different ensembles and constraints.
A few widely-used specializations are provided for all four aspects, while customized ones can be added easily.

The main aim of the `flory` package is to find the coexisting phases in multicomponent mixtures conveniently, while also being efficient enough for sampling entire phase diagrams in high dimensions.
The task of finding coexisting phases is completed by finders, which are initialized once and then reused for multiple parameters to reduce overhead.
The core methods of the package are just-in-time (JIT) compiled using numba [@lam2015Numba] to achieve high performance.
Beside the core function of finding coexisting phases, `flory` also includes convenience tools, e.g., to analyze thermodynamic properties.

# Methods

The core part of the `flory` package is the finder for coexisting phases, which can be reused when the number $N_\mathrm{C}$ of components is kept fixed.
This design moves a significant overhead to the creation of the solver, which can be amortized in many tasks, e.g., when a phase diagram is sampled.
We illustrate this idea in the following code, which finds the two coexisting phases of a symmetric Flory-Huggins binary mixture with $\chi=4$:
```python
fh = flory.FloryHuggins(2, chis=[[0, 4.0], [4.0, 0]])
ensemble = flory.CanonicalEnsemble(2, phi_means=[0.5, 0.5])
finder = flory.CoexistingPhasesFinder(fh.interaction, fh.entropy, ensemble)
phases = finder.run().get_clusters()
```
Here, `FloryHuggins` represents the seminal Flory-Huggins free energy that creates the interaction `FloryHugginsInteraction` and the entropy `IdealGasEntropy` simultaneously, and provides tools for analyzing coexisting phases.
By updating the interaction, we can then obtain the coexisting phases of another symmetric binary mixture with $\chi=3.5$:
```python
fh.chis = [[0, 3.5], [3.5, 0]]
finder.set_interaction(fh.interaction)
phases = finder.run().get_clusters()
```
This procedure can be repeated to sample an entire phase diagrams.
Moreover, we could vary the type of interaction by initializing a different class or modifying the existing one, and we could similarly change the entropy, ensemble, and constraints.
Customized specialization of all four aspects can be easily implemented by deriving from the provided base classes.

The `flory` package is designed to deliver high performance, which is achieved by just-in-time compilation using numba [@lam2015Numba].
To support different forms of the free energy $f$, the core method in the finder is designed to be general.
The finder fetches compiled instances of the interaction, entropy, ensemble, and constraints, where case-specific codes are inserted as methods.
These methods are also compiled for performance, using the JIT class feature from numba [@lam2015Numba].

The `flory` package adopts state-of-the-art numerical methods to determine coexisting phases.
The main idea is to represent the system by many independent compartments, which can exchange particles and volumes, obeying total constraints [@zwicker2022Evolved].
The `flory` package then minimizes the full free energy $\bar f$ instead of directly solving the corresponding coexistence conditions.
At the free energy minimum, compartments may share similar compositions, which the package then cluster to obtain unique phases.
This strategy circumvents the typical challenge of multiple local minima and it avoids iterating over all possible phase counts $N_\mathrm{P}$.
To improve performance, the `flory` package implements the improved Gibbs ensemble method developed recently [@qiang2024Scaling].
This method redistributes components across all compartments simultaneously, guided by a set of conjugate variables, such that the total computation cost per step only scales linearly with the number of compartments.
In summary, the `flory` package can typically obtain the equilibrium coexisting states even in systems with many interacting components.

# References