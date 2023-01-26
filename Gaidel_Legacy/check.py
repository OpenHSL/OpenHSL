import matplotlib.pyplot as plt
import numpy as np

#check npy file
a = np.load('cube.npy')
plt.imshow(a[10, :, :], cmap="gray")
plt.show()