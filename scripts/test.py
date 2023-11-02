import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
from tqdm import tqdm
import scipy.optimize as opt

from main import Percolation2D

size = 10
p = 0.5

percolation2d = Percolation2D()
lattice, max_cluster, labeled_lattice = percolation2d.generate(size, p)

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(lattice)
ax2.imshow(labeled_lattice)
ax3.imshow(max_cluster)
plt.show()