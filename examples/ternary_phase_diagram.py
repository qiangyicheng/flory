import matplotlib.pyplot as plt
import numpy as np

import flory

num_comp = 3
chis = [[0.0, 2.4, 3.2], [2.4, 0.0, 2.8], [3.2, 2.8, 0.0]]
# we guess this point is a three-phase coexistence point
phi_means = [0.33, 0.33, 0.34]

free_energy = flory.FloryHuggins(num_comp, chis)
interaction = free_energy.interaction
entropy = free_energy.entropy
ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
options = {
    "num_part": 8,
    "progress": False,
    "max_steps": 10000000,  # disable progress bar, allow more steps
}
finder = flory.CoexistingPhasesFinder(interaction, entropy, ensemble, **options)

# determine the three phase coexistence
phases = finder.run().get_clusters()
assert phases.volumes.shape[0] == 3
p3_phis = phases.fractions
p3_center = np.mean(p3_phis, axis=0)
p3_edges = [p3_phis[[1, 2]], p3_phis[[0, 2]], p3_phis[[0, 1]]]

# function that scan the 2-phase coexistence until the boundary is reached
def find_p2_boundaries(
    init_tie: np.ndarray,
    start_point: np.ndarray,
    finder: flory.CoexistingPhasesFinder,
    scan_step: float = 0.02,
    scan_step_min: float = 0.000001,
):
    internal_finder = flory.CoexistingPhasesFinder(
        interaction, entropy, ensemble, **options
    )

    internal_finder.reinitialize_from_omegas(
        finder.omegas.copy()
    )  # use previous internal fields to accelerate

    previous = start_point
    current_tie = init_tie
    step = scan_step

    ties = [init_tie]
    while True:
        tie_center = current_tie.mean(axis=0)
        tie_dir = current_tie[1] - current_tie[0]
        tie_dir = tie_dir / np.linalg.norm(tie_dir)
        next_dir = tie_center - previous
        next_dir = next_dir - np.dot(next_dir, tie_dir) * tie_dir
        next_dir = next_dir / np.linalg.norm(next_dir)
        next_phis = tie_center + step * next_dir
        
        ensemble.phi_means = np.array(next_phis)
        internal_finder.set_ensemble(ensemble)
        backup_omegas = internal_finder.omegas.copy()
        phases = internal_finder.run().get_clusters()
        
        if phases.volumes.shape[0] == 2 and np.all(phases.fractions > 0):
            ties.append(phases.fractions)
            previous = tie_center
            current_tie = phases.fractions
        else:
            internal_finder.reinitialize_from_omegas(backup_omegas)
            if step > scan_step_min:
                step /= 2
            else:
                break

    return np.array(ties)


p2_ties = [find_p2_boundaries(edge, p3_center, finder) for edge in p3_edges]

# plot the allowed region
plt.plot([0, 1, 0, 0], [1, 0, 0, 1], c="black")

for ties in p2_ties:
    # plot the phase boundaries
    plt.plot(ties[:, 0, 0], ties[:, 0, 1], c="brown")
    plt.plot(ties[:, 1, 0], ties[:, 1, 1], c="brown")
    for tie in ties:
        # plot the tie lines
        plt.plot(tie[:, 0], tie[:, 1], c="gray")

# plot the 3-phase region
plt.plot(p3_phis[[0, 1, 2, 0], 0], p3_phis[[0, 1, 2, 0], 1], c="blue")

plt.xlabel("$\\phi_A$")
plt.ylabel("$\\phi_B$")
plt.savefig(__file__ + ".jpg")
