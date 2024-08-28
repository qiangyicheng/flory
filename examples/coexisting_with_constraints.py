import numpy as np
import flory

num_comp = 5
chis = np.zeros((num_comp, num_comp))
chis[0][1] = -7.0
chis[1][0] = -7.0
phi_means = [0.10, 0.10, 0.09, 0.1 * 0.872413793103448, 0]
phi_means[4] = 1.0 - np.sum(phi_means)
sizes = [10.0, 10.0, 1.0, 1.0, 1.0]

Cs = [0.9, -0.872413793103448, -1, 1, 0]
Ts = 0

fh = flory.FloryHuggins(num_comp, chis, sizes=sizes)
ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
constraint = flory.LinearLocalConstraint(num_comp, Cs, Ts)

finder = flory.CoexistingPhasesFinder(
    fh.interaction,
    fh.entropy,
    ensemble,
    [constraint],
    max_steps=1000000,
    progress=True,
)
volumes, phis = finder.run()

with open(__file__ + ".out", "w") as f:
    print("Volumes:", volumes, file=f)
    print("Compositions:", phis, file=f)