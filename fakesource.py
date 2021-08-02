import numpy as np
import matplotlib.pyplot as plt

I0 = -20
sigma = 500

sidelength=2000
def gaussian(x,y,I, sig):
    return (-1)*I * np.e**(-(x**2 + y**2)/(2*sig**2))

def solveGaussian(I , width):
    return I/(2*np.pi*width**2)
print(I0/(2*np.pi*sigma**2))

x_arr = np.arange(-1*sidelength, sidelength, 1)
y_arr = np.arange(-1*sidelength, sidelength, 1)

z_arr = []
i = (-1)*sidelength
j = (-1)*sidelength


for i in x_arr:
    z = []
    for j in y_arr:
        z.append(gaussian(i,j, solveGaussian(I0, sigma), sigma))
    z_arr.append(z)

plt.imshow(z_arr, extent=[(-1)*sidelength, sidelength, (-1)*sidelength, sidelength])
fig = plt.gcf()
ax = fig.gca()

ax.add_patch(plt.Circle((0, 0), sigma, color='black', linestyle='dashed', fill = False))
plt.colorbar()
plt.savefig('fakesource.png')
plt.show()