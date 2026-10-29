import matplotlib as mpl
import matplotlib.pyplot as plt

import flory

chis = [[3.27, -0.34, 0], [-0.34, -3.96, 0], [0, 0, 0]]
phi_means = [0.16, 0.55, 0.29]
sizes = [2.0, 2.0, 1.0]

#generate the data here
resulting_phases = flory.find_coexisting_phases(3, chis, phi_means, sizes=sizes)

"""
Now, we will highlight three different ways for how to use the plot() method.
"""

"""
First example

Simply invoking the plot() method will generate a default plot with axes labels, ticks, and a colorbar set with default values.
"""
resulting_phases.plot()
plt.show()

"""
Second example

If the user wants to customize the plot, they can do so by passing in a matplotlib axes object
and setting the labels, ticks, and colorbar themselves. For example:
"""
fig, ax = plt.subplots()
cmap = "Blues" #set the colormap to Blues (monochrome)

im = resulting_phases.plot(ax=ax, cmap=cmap) #the image is returned so that the user can set the colorbar themselves, if they like
ax.set_xlabel("Phase")
ax.set_ylabel("Component") 
ax.set_title("My Custom Plot of Coexisting Phases")

#Here, we now add a custom colorbar
fig.colorbar(im, cmap=cmap, ax=ax, label="Volume Fraction")
plt.show()

"""
Third example

We can also set the min and max values for the colorbar
"""
resulting_phases.plot(vmin=0, vmax=0.5) 
plt.show()
