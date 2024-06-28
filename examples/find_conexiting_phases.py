import mcmp

chis = [[0, 4.0], [4.0, 0]]
phi_means = [0.5, 0.5]

volumes, phis = mcmp.cpfinder(chis, phi_means, 16)

print(volumes)
print(phis)
