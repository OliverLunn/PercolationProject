import numpy as np
import matplotlib.pyplot as plt

from main import Percolation2D

perc = Percolation2D()
L=200
p=0.59274605079210 
lattice = perc.lattice_random(L,0.59274605079210)
labeled_lattice = perc.cluster_search(lattice)
max_cluster = perc.max_cluster(L, labeled_lattice)

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(max_cluster,cmap='binary')
ax1.set_xticks([])
ax1.set_yticks([])

ax2.imshow(max_cluster,cmap='binary')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim(50,150)
ax2.set_ylim(50,150)

ax3.imshow(max_cluster,cmap='binary')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlim(75,125)
ax3.set_ylim(75,125)

plt.tight_layout()
plt.show()