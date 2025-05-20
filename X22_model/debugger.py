import numpy as np
from scipy.interpolate import interp2d, RegularGridInterpolator

x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 50)

xx, yy = np.meshgrid(x, y)

z = np.sin(xx**2 + yy**2)
f = interp2d(x, y, z, kind='linear')

x_new = np.linspace(0, 1, 100)
y_new = np.linspace(0, 1, 500)
xx_new, yy_new = np.meshgrid(x_new, y_new)

# z_new = f(x_new, y_new)
# print(z_new.shape)  # (500, 100)

f = RegularGridInterpolator((x, y), z.T)

z_new = f((xx_new, yy_new)).T


# z_new = r((x_new, y_new))
print(z_new.shape)  # (500, 100)