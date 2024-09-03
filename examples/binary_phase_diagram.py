import matplotlib.pyplot as plt
import numpy as np

import flory

chi_start = 5.0
chi_end = 1.0

num_comp = 2
chis = [[0.0, 0.0], [0.0, 0.0]]
phi_means = [0.5, 0.5]

free_energy = flory.FloryHuggins(num_comp, chis)
interaction = free_energy.interaction
entropy = free_energy.entropy
ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
finder = flory.CoexistingPhasesFinder(
    interaction,
    entropy,
    ensemble,
    progress=False,
)


line_chi = []
line_l = []
line_h = []
for chi in np.arange(chi_start, chi_end, -0.1):  # scan chi from high value to low value
    interaction.chis = np.array([[0, chi], [chi, 0]])  # set chi matrix of the finder
    finder.set_interaction(interaction)
    phases = finder.run().get_clusters()  # get coexisting phases
    if phases.fractions.shape[0] == 1:  # stop scanning if no phase separation
        break
    phi_h = phases.fractions[0, 0]  # extract the volume fraction of component 0 in phase 0
    phi_l = phases.fractions[1, 0]  # extract the volume fraction of component 0 in phase 1
    line_chi.append(chi)
    line_l.append(phi_l)
    line_h.append(phi_h)

plt.plot(line_l, line_chi, c="black")
plt.plot(line_h, line_chi, c="black")
plt.xlabel("$\\phi$")
plt.ylabel("$\\chi$")
plt.savefig(__file__ + ".jpg")
