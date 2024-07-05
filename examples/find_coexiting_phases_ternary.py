import flory

chis = [[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
phi_means = [0.16, 0.55, 0.29]
sizes = [2.0, 2.0, 1.0]

volumes, phis = flory.find_coexisting_phases(chis, phi_means, 16, sizes = sizes)

with open(__file__ + ".out", "w") as f:
    print("Volumes:", volumes, file=f)
    print("Compositons:", phis, file=f)