import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from main import Percolation2D
def func(x,gamma,p_c):
    return np.abs(x-p_c)**(gamma)

percolation2D = Percolation2D()
size = 100
probs=np.arange(0.4,0.9,0.001)
runs=10
avg_size = np.zeros((runs,len(probs)))
for r in range(runs):
    i=0
    for p in probs:
        lattice = percolation2D.lattice_random(size, p)
        labeled_lattice = percolation2D.cluster_search(lattice)
        avg_size[r,i] = percolation2D.average_cluster_size(lattice)
        i+=1
avg_size = avg_size/(size**2)
plt.plot(probs,np.average(avg_size,axis=0))
plt.show()



#ppot,pcov = opt.curve_fit(func,probs,np.average(avg_size,axis=0))