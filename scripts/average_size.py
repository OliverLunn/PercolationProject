import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

from main import Percolation2D

def func(x,m,c):
    return m*x + c

percolation2D = Percolation2D()
size = 100
probs=np.arange(0.4,0.58,0.001)
runs=50
avg_size = np.zeros((runs,len(probs)))
for r in tqdm(range(0,runs)):
    i=0
    lattice = percolation2D.rand_lattice(size)
    for p in probs:
        lattice_occupied= percolation2D.occupied(lattice,size,p)
        avg_size[r,i] = percolation2D.average_cluster_size(lattice_occupied)
        i+=1

avg_size = avg_size/(size**2)
plt.figure()
ax = plt.axes()
p_c = 0.59274605079210
ax.plot(np.log(np.abs(probs-p_c)),np.log(np.average(avg_size,axis=0)))
#np.savetxt(f'{size}x{size} lattice average size data over {runs} runs.txt',avg_size)



ppot,pcov = opt.curve_fit(func,np.log(np.abs(probs-p_c)),np.log(np.average(avg_size,axis=0)))
ax.plot(np.log(np.abs(probs-p_c)),func(np.log(np.abs(probs-p_c)),*ppot),color='black',label='fit')
print(ppot)
plt.show()