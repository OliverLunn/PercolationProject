import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

from main import Percolation2D

def func(x,m,c):
    return m*x + c

percolation2D = Percolation2D()
sizes = [50,75,100,150,200,250]
probs=np.linspace(0.3,0.7,100)
S = np.zeros((len(sizes),len(probs)))
j=0
for size in tqdm(sizes):
    runs=int(500*25/size)
    for r in tqdm(range(0,runs)):
        i=0
        lattice = percolation2D.rand_lattice(size)
        for p in probs:
            lattice_occupied= percolation2D.occupied(lattice,size,p)
            S[j,i] =S[j,i] + percolation2D.average_cluster_size(lattice_occupied)
            i+=1
    S[j,:] = S[j,:]/((size**2)*runs)
    j+=1

fig,(ax1)=plt.subplots(1)
gammas = np.zeros(len(S))
errs = np.zeros(len(S))
p_c = 0.59274605079210
indx = np.max(np.where(probs<p_c))
colors=['red','orange','yellow','green','blue','purple']
for j in range(len(sizes)):
    
    ax1.plot(probs,S[j,:],color = colors[j],label=f'L={int(sizes[j])}')

    ppot,pcov = opt.curve_fit(func,np.log(np.abs(probs[0:indx]-p_c)),np.log(S[j,0:indx]))
    errs[j] = np.sqrt(np.diag(pcov))[0]
    gammas[j] = ppot[0]
np.savetxt(f'data\square lattice average size data.txt',S)
np.savetxt('data\gamma estimation square.txt',np.vstack((sizes,gammas,errs)))
ax1.vlines(p_c,np.min(S[j,:]),np.max(S[j,:]),color='black',linestyle='--',label='P_c')
ax1.legend(fontsize="18")
ax1.tick_params(axis="x", labelsize=18)
ax1.tick_params(axis="y", labelsize=18)
ax1.set_xlabel('$P$',fontsize="22")
ax1.set_ylabel('$S(p,L)$',fontsize="22")
plt.show()