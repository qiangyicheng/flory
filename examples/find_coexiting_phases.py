import flory

chis = [[0, 4.0], [4.0, 0]]
phi_means = [0.5, 0.5]

phases = flory.find_coexisting_phases(2, chis, phi_means)

with open(__file__ + ".out", "w") as f:
    print("Volumes:", phases.volumes, file=f)
    print("Compositions:", phases.fractions, file=f)