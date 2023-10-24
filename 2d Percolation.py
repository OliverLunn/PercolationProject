import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

size = 100 
p = 0.6 
rand_lattice = np.random.rand(size,size) 
occupied_lattice = rand_lattice<p 
labeled_lattice,num = label(occupied_lattice)
count = np.bincount(np.reshape(labeled_lattice,size*size))
count[0] = 0
max_cluster_id = np.argmax(count)
lattice = np.where(labeled_lattice == max_cluster_id,0.5,0)
lattice = np.where(occupied_lattice == True, lattice+0.5,lattice)
plt.imshow(lattice,cmap='gray')
plt.show()