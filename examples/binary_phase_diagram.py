import flory
import numpy as np
import matplotlib.pyplot as plt

chi_start = 5.0
chi_end = 1.0

chis = [[0., 0.], [0., 0.]]
phi_means = [0.5, 0.5]

finder = flory.CPFinder(chis, phi_means, 16, progress=False)

line_chi = []
line_l = []
line_h = []
chi_ODT = 0.0
for chi in np.arange(chi_start, chi_end, -0.1):
    chis = chis = [[0, chi], [chi, 0]]
    finder.chis = chis
    volumes, phis = finder.run()
    phi_h = phis[0, 0]
    phi_l = phis[0, 1]
    if np.abs(phi_l - phi_h) < 1e-3:
        chi_ODT = chi
        break
    line_chi.append(chi)
    line_l.append(phi_l)
    line_h.append(phi_h)

plt.plot(line_l, line_chi)
plt.plot(line_h, line_chi)
plt.show()
