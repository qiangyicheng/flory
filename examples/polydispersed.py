import flory
import numpy as np

num_feat = 2
chis_feat = [[0, 4.0], [4.0, 0]]
phi_means = [0.2, 0.3, 0.2, 0.3]
sizes = [1, 2, 1, 2]
num_comp_per_feat = [2, 2]
num_comp = np.sum(num_comp_per_feat)

interaction = flory.FloryHugginsBlockInteraction(num_feat, chis_feat, num_comp_per_feat)
entropy = flory.IdealGasPolydispersedEntropy(num_feat, sizes, num_comp_per_feat)
ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
finder = flory.CoexistingPhasesFinder(interaction, entropy, ensemble)

phases = finder.run().get_clusters()

with open(__file__ + ".out", "w") as f:
    print("Volumes:", phases.volumes, file=f)
    print("Compositions:", phases.fractions, file=f)


