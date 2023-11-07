import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

from main import Percolation2D

def func(x,m,c):
    return m*x + c

percolation2D = Percolation2D()
sizes = [75,100,150,200,250,300]
probs=np.linspace(0.4,0.7,75)
avg_size = np.zeros((len(sizes),len(probs)))
j=0
for size in tqdm(sizes):
    runs=int(1000*25/size)
    for r in tqdm(range(0,runs)):
        i=0
        lattice = percolation2D.rand_lattice(size)
        for p in probs:
            lattice_occupied= percolation2D.occupied(lattice,size,p)
            avg_size[j,i] =avg_size[j,i] + percolation2D.average_cluster_size(lattice_occupied)
            i+=1
    avg_size[j,:] = avg_size[j,:]/((size**2)*runs)
    j+=1

fig,(ax1)=plt.subplots(1)

p_c = 0.59274605079210
gammas = np.zeros(len(sizes))
errs = np.zeros(len(sizes))
colors=['red','orange','yellow','green','blue','purple']
indx = np.max(np.where(probs<p_c))
for j in range(len(sizes)):
    ax1.plot(probs,avg_size[j,:],color=colors[j],label=f'L={int(sizes[j])}')

    #ax2.plot(np.log(np.abs(probs[0:indx]-p_c)),np.log(avg_size[j,0:indx]))
    #np.savetxt(f'{size}x{size} lattice average size data over {runs} runs.txt',avg_size)

    ppot,pcov = opt.curve_fit(func,np.log(np.abs(probs[0:indx]-p_c)),np.log(avg_size[j,0:indx]))
    #ax2.plot(np.log(np.abs(probs[0:indx]-p_c)),func(np.log(np.abs(probs[0:indx]-p_c)),*ppot),color='black',label='fit')
    gammas[j] = ppot[0]
    errs[j] = np.sqrt(np.diag(pcov))[0]

print(gammas,errs)
print(np.average(gammas),np.average(errs))
ax1.set_xlabel('$p$')
ax1.set_ylabel('$S(p,L)$')
ax1.legend()
plt.show()