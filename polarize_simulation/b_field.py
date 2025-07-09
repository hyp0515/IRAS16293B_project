import numpy as np
import matplotlib.pyplot as plt

# Create radial and angular grid
r_grid = np.linspace(0, 100, 50)  # radial grid in AU
theta_grid = np.linspace(0, 2 * np.pi, 50)  # angular grid in radians

# Create a meshgrid
R, Theta = np.meshgrid(r_grid, theta_grid)

XX = R * np.cos(Theta)  # X coordinates in Cartesian
YY = R * np.sin(Theta)  # Y coordinates in Cartesian


alpha = 0.004


# Define magnetic field components in polar coordinates
Br      = alpha*(R**2)*(np.sin(Theta)**2)*np.cos(Theta)*np.exp(-alpha*(R**2)*(np.cos(Theta)**2))+np.cos(Theta)   # radial component
Btheta  = alpha*(R**2)*(np.cos(Theta)**2)*np.sin(Theta)*np.exp(-alpha*(R**2)*(np.cos(Theta)**2))-np.sin(Theta)   # angular component




# Convert polar to Cartesian coordinates for plotting
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# Convert polar vector components to Cartesian vector components
Bx = Br * np.cos(Theta) - Btheta * np.sin(Theta)
By = Br * np.sin(Theta) + Btheta * np.cos(Theta)

# Plot using quiver
plt.figure(figsize=(8, 8))
plt.quiver(Y, X, By, Bx, scale=50, width=0.002, color='blue')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Magnetic Field Vectors in Polar Coordinates')
plt.axis('equal')  # To ensure circles look circular
plt.grid(True)
plt.show()