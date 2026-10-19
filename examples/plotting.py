import matplotlib.pyplot as plt

import flory

chis = [[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
phi_means = [0.16, 0.55, 0.29]
sizes = [2.0, 2.0, 1.0]

resulting_phases = flory.find_coexisting_phases(3, chis, phi_means, sizes=sizes)

fig, ax = resulting_phases.plot(figtitle = "Resulting Phases", figtitle_size = 15)

savefig = False
if savefig:
    fig.savefig("example_phases_plot.png", dpi=300, bbox_inches="tight")

plt.show()