import numpy as np

import flory

num_comp = 3
chis = (1 - np.identity(num_comp)) * 5

mus = [-1, 0.0, 0.1]

Cs = [[1, 1, 0]]
Ts = [0.4]

fh = flory.FloryHuggins(num_comp, chis)
ensemble = flory.GrandCanonicalEnsemble.from_chemical_potential(num_comp, mus)
constraint = flory.LinearGlobalConstraint(num_comp, Cs, Ts)

finder = flory.CoexistingPhasesFinder(
    fh.interaction,
    fh.entropy,
    ensemble,
    constraint,
    random_std=1.0,  # use less aggressive randomness to avoid rapid dying of compartments
    progress=True,
    tolerance=1e-12
)
phases = finder.run().get_clusters()

with open(__file__ + ".out", "w") as f:
    print("Volumes:", phases.volumes, file=f)
    print("Compositions:", phases.fractions, file=f)
