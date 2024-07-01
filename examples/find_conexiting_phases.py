import flory

chis = [[0, 4.0], [4.0, 0]]
phi_means = [0.5, 0.5]

volumes, phis = flory.coexisting_phases_finder(chis, phi_means, 16)

print(volumes)
print(phis)
