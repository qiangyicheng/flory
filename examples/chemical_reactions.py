import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import flory

# Setup 3-component system with two reaction components A and B and a third component C,
# which does not undergo reaction, but instead phase separates
num_comp = 3
stoichiometric_matrix = [1, -1, 0]
phi_means = [0.4, 0.4, 0.2]  # mean fraction of all components

# setup thermodynamics
chis = [[0, 0, 4], [0, 0, 4], [4, 4, 0]]
fh = flory.FloryHuggins(num_comp, chis)
sizes = [5, 1, 1]

# find the coexisting phases
phases = flory.find_coexisting_phases(
    num_comp,
    chis,
    phi_means=phi_means,
    stoichiometric_matrix=stoichiometric_matrix,
    sizes=sizes,
)

# print the fractions (one phase enriched in C and one more A than B)
print(phases.fractions)
