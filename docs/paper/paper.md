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
date: 11 August 2024
bibliography: paper.bib
---

# Summary

Phase separation, an intrinsic property of mixtures, is widely observed in many scenarios, ranging from the simple demixing of oil and water to the condensation of biomolecules in cells.
In multicomponent mixtures, phase separation leads to the coexistence of multiple phases.
Finding these coexisting phases is a general question in many fields such as chemical engineering[@lukas2007Computational] and soft matter physics [@jacobs2023Theory].
There are several strategies to theoretically predict the coexisting phases [@shrinivas2021Phase], including direct spatial-resolved simulation[@shrinivas2021Phase], convex hulling of the free energy landscape [@mao2019Phase], solving the balance equations [@zwicker2022Evolved] and the free energy minimization [@lukas2007Computational].
However, the existing methods of these strategies are usually designed for small number of components, whereas there's a surge in interest in the study of multi-component cases recently.
With many components, finding coexisting phases becomes much more challenging, mainly due to the high dimensional nature of the phase space and the couping of many nonlinear equations.

The `flory` python package presented in this paper provides convenient tools for the researchers to find the coexisting phases of a broad range of multicomponent mixtures, especially the mixtures with many components.
The package focuses on the coexisting phases in thermodynamically large systems, where the interfaces between phases become negligible.
The package finds the coexisting phases by minimization the following average free energy density,

$$\bar{f}({N_\mathrm{P}}, \{J_p\}, \{\phi_{p,i}\}) = \sum_{p=1}^{{N_\mathrm{P}}} J_p f(\{\phi_{p,i}\}) \; ,$$

where $N_\mathrm{C}$ is the number of components, $N_\mathrm{P}$ is the number of phases, $J_p$ is the volume fraction of the phase $p$ and $\phi_{p,i}$ is the volume fraction of the component $i$ within the phase $p$.

To achieve flexibility, `flory` expresses any mixture by four orthogonal aspects: interaction, entropy, ensemble and constraints.
The package only imposes limits on the entropy part that is crucial for the core algorithm, while keeping other three aspects arbitrary.
By combining these four aspects, `flory` supports a broad range of the forms of the free energy $f$ with different ensembles and constraints.
A few widely-used specializations are provided for all four aspects, while customized ones can be easily added.

The main aim of `flory` package is to find the coexisting phases in multicomponent mixtures in a convenient way that is also efficient enough for constructing phase diagrams or sampling phase diagrams in high dimensions.
The task of finding coexisting phases is completed by finders.
To reduce overhead, finder instance can be created and reused for multiple sets of parameters.
The core methods of the package are just-in-time (JIT) compiled using numba [@lam2015Numba] to achieve high performance, where the finder part is designed to be general for all supported forms of free energy.
`flory` package also includes tools for analyzing the thermodynamic properties of the coexisting phases.

# Methods

The main part of the `flory` package is the finder for coexisting phases that can be reused on different parameters once created when the system size is fixed.
This design separates most of the overheads to the creation of the solver, which can be amortized in many tasks such as constructing or sampling the phase diagram of a given mixture.
For instance, the following code finds the two coexisting phases of a symmetric Flory-Huggins binary mixture with $\chi=4$:
```python
fh = flory.FloryHuggins(2, chis = [[0, 4.0], [4.0, 0]])
ensemble = flory.CanonicalEnsemble(2, phi_means = [0.5, 0.5])
finder = flory.CoexistingPhasesFinder(fh.interaction, fh.entropy, ensemble)
phases = finder.run().get_clusters()
```
Here `FloryHuggins` represents the widely-used Flory-Huggins free energy that creates the interaction `FloryHugginsInteraction` and the entropy `IdealGasEntropy` simultaneously, and provides tools for analyzing coexisting phases. 
Then the coexisting phases of another symmetric binary mixture with $\chi=3.5$ matrix can be obtained by updating the interaction:
```python
fh.chis = [[0, 3.5], [3.5, 0]]
finder.set_interaction(fh.interaction)
phases = finder.run().get_clusters()
```
Such procedures can be repeated to complete certain tasks such as generating or sampling the phase diagrams.
A different interaction type can be used to update the finder as well.
Entropy, ensemble and constraints can also be set or updated similarly.  
Similar to interaction and entropy, `flory` also provides common cases for ensemble and constraints.
Customized specialization for all four aspects can be easily implemented by deriving from the provided base classes.

The `flory` package is designed to deliver high performance.
All the key methods are just-in-time compiled using numba [@lam2015Numba].
To support different forms of free energy, the core method in the finder is designed to be general.
The finder fetches compiled instances of interaction, entropy, ensemble and constraints, where case-specific codes are inserted as methods.
These methods are also compiled for performance, using the JIT class feature from numba [@lam2015Numba].

The `flory` package adopts the state-of-art numerical methods to determine the coexisting phases, based on the idea of compartments.
In each instance, package starts with many initial compartments that exchange particles and volumes, whose number is fixed [@zwicker2022Evolved].
The `flory` package then minimizes the full free energy instead of solving the coexistence conditions.
At the free energy minimum, compartments may share similar compositions, which the package then cluster to obtain unique phases.
This strategy relieves the typical problem of multiple local minimum and avoids iterate over different phase number $N_\mathrm{P}$ in calculation.
To reduce the influence of using many compartments on performance, the `flory` package implements the improved Gibbs ensemble method developed recently [@qiang2024Scaling].
This method redistributes components across all compartments guided by a set of conjugate variables, such that the total computation cost per step only scales linearly with the number of compartments.
Therefore, the `flory` package can promisingly obtain the equilibrium coexisting states even though this is not guaranteed theoretically. 

# References