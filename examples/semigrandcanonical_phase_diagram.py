import sys

# my local path
package_path = "/Users/skoestler/"

sys.path.insert(0, package_path + "flory")
sys.path

import numpy as np
import matplotlib.pyplot as plt
import flory

num_comp = 4
# components in order D, A, B, S(olvent)
chis = [[0, 1.9, 0.5, 5], [1.9, 0, 0, -1.6], [0.5, 0, 0, 0.9], [5, -1.6, 0.9, 0]]

fh = flory.FloryHuggins(num_comp, chis)

# set up arrays to save the volume fractions of D and exchange chemical potentials of A and B
phi_D_dense = np.array([])
phi_D_dilute = np.array([])
mu_A = np.array([])
mu_B = np.array([])

for muA in np.arange(-2, 4, 0.3):
    for muB in np.arange(-2, 1, 0.15):
        for phiD in np.arange(0.05, 0.5, 0.05):
            Constraints = [phiD, np.exp(muA), np.exp(muB), 1.0] # setting the solvent chemical potential to 0 without loss of generality
            ensemble = flory.SemiGrandCanonicalEnsemble(num_comp, [True, False, False, False], Constraints) # component D is treated canonically, while A and B are coupled to a reservoir
            finder = flory.CoexistingPhasesFinder(
                fh.interaction,
                fh.entropy,
                ensemble,
                random_std=1.0,
                progress=False,
                tolerance=1e-12,
            )
            phases = finder.run().get_clusters()

            if len(phases.fractions) == 2:
                if phases.fractions[0][0] > phases.fractions[1][0]:
                    phi_D_dense = np.append(phi_D_dense, np.array([phases.fractions[0][0]]))
                    phi_D_dilute = np.append(phi_D_dilute, np.array([phases.fractions[1][0]]))
                else:
                    phi_D_dense = np.append(phi_D_dense, np.array([phases.fractions[1][0]]))
                    phi_D_dilute = np.append(phi_D_dilute, np.array([phases.fractions[0][0]]))
                mu_A = np.append(mu_A, np.array([muA]))
                mu_B = np.append(mu_B, np.array([muB]))
                break

# plot the dense branch of the semi-grand-canonical phase diagram
fig, ax = plt.subplots(figsize=(5,4))
a = ax.scatter(mu_A, mu_B, c=phi_D_dense, cmap='viridis', s=40, vmin=0, vmax=1)
cbar = plt.colorbar(a, shrink=0.9)
cbar.set_label(r'$\phi_\mathrm{D}$', fontsize=18)
ax.set_xlabel(r'$\bar\mu_A$', fontsize=18)
ax.set_ylabel(r'$\bar\mu_B$', fontsize=18)
ax.set_xlim(-2,4)
ax.set_ylim(-2,1)
fig.savefig('semigrandcanonical_phase_diagram.py.png', dpi=300, bbox_inches='tight')