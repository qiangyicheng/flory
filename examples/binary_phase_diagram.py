import flory
import numpy as np
import matplotlib.pyplot as plt

chi_start = 5.0
chi_end = 1.0

chis = [[0.0, 0.0], [0.0, 0.0]]
phi_means = [0.5, 0.5]

finder = flory.CoexistingPhasesFinder(
    chis, phi_means, 16, progress=False  # disable progress bar
)  # create a finder

line_chi = []
line_l = []
line_h = []
for chi in np.arange(chi_start, chi_end, -0.1):  # scan chi from high value to low value
    finder.chis = [[0, chi], [chi, 0]]  # set chi matrix of the finder
    volumes, phis = finder.run()  # get coexisting phases
    if phis.shape[0] == 1:  # stop scanning if no phase separation
        break
    phi_h = phis[0, 0]  # extract the volume fraction of component 0 in phase 0
    phi_l = phis[1, 0]  # extract the volume fraction of component 0 in phase 1
    line_chi.append(chi)
    line_l.append(phi_l)
    line_h.append(phi_h)

plt.plot(line_l, line_chi, c="black")
plt.plot(line_h, line_chi, c="black")
plt.xlabel("$\\phi$")
plt.ylabel("$\\chi$")
plt.savefig(__file__ + ".jpg")
