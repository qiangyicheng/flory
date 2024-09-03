import flory

chis = [[0, 4.0], [4.0, 0]]
phi_means = [0.5, 0.5]

fh = flory.FloryHuggins(2, chis)
ensemble = flory.CanonicalEnsemble(2, phi_means)
finder = flory.CoexistingPhasesFinder(fh.interaction, fh.entropy, ensemble)

phases = finder.run().get_clusters()

with open(__file__ + ".out", "w") as f:
    print("Volumes:", phases.volumes, file=f)
    print("Compositions:", phases.fractions, file=f)
