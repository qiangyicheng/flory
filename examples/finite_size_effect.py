import matplotlib.pyplot as plt
import numpy as np

import flory

N_comp = 8

# create a random chi matrix
chi_mean = 0
chi_std = 8
rng = np.random.default_rng(2333)
chis = rng.normal(chi_mean, chi_std, (N_comp, N_comp))
chis = 0.5 * (chis + chis.T)
chis *= 1.0 - np.identity(N_comp)

phi_means = np.full(N_comp, 1.0 / N_comp)  # set a symmetric composition

line_N_compartment = []
line_N_phase = []
for N_compartment in range(4, 16, 2): # use different compartment number
    phases = flory.find_coexisting_phases(
        N_comp, chis, phi_means, num_part = N_compartment, progress=True
    )
    line_N_compartment.append(N_compartment)
    line_N_phase.append(phases.volumes.shape[0])

plt.plot(line_N_compartment, line_N_phase, c="black")
plt.xlabel("number of compartments")
plt.ylabel("number of phases")
plt.savefig(__file__ + ".jpg")
