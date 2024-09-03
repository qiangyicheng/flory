import flory

chis = [[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
phi_means = [0.16, 0.55, 0.29]
sizes = [2.0, 2.0, 1.0]

phases = flory.find_coexisting_phases(3, chis, phi_means, sizes = sizes)

with open(__file__ + ".out", "w") as f:
    print("Volumes:", phases.volumes, file=f)
    print("Compositions:", phases.fractions, file=f)