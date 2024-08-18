import flory

chis = [[0, 4.0], [4.0, 0]]
phi_means = [0.5, 0.5]

volumes, phis = flory.find_coexisting_phases(2, chis, phi_means)

with open(__file__ + ".out", "w") as f:
    print("Volumes:", volumes, file=f)
    print("Compositions:", phis, file=f)