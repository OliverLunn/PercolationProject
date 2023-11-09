import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as opt

from main import Percolation2D

percolation2D = Percolation2D()

sizes = [50,75,100,150,200,250]
probs=np.linspace(0.4,0.75,50)

perc_prob = np.zeros((len(sizes),len(probs)))

j=0
for size in sizes:
    runs=int(2000*25/size)
    for r in tqdm(range(0,runs)):
        i=0
        lattice = percolation2D.rand_lattice(size)
        for p in probs:
            lattice_occupied= percolation2D.occupied(lattice,size,p)
            perc_prob[j,i] = perc_prob[j,i] + percolation2D.percolation_probability(lattice_occupied)
            i+=1
    
    perc_prob[j,:] = perc_prob[j,:]/runs
    j+=1
np.savetxt('data/percolation probabilty.txt',perc_prob)

fig,(ax1) = plt.subplots(1,1)
colors=['red','orange','yellow','green','blue','purple']
pcs = np.zeros(len(sizes))
for j in range(0,len(sizes)):
    ax1.plot(probs,perc_prob[j,:],label = f'L={int(sizes[j])}',color=colors[j])

    ipc = np.argmax(perc_prob[j,:]>0.5) # Find first value where Perc_prob>0.5
    # Interpolate from ipc-1 to ipc to find intersection
    ppc = probs[ipc-1] + (0.5-perc_prob[j,ipc-1])*\
        (probs[ipc]-probs[ipc-1])/(perc_prob[j,ipc]-perc_prob[j,ipc-1])
    pcs[j]=ppc
    ax1.scatter(ppc,0.5,color = 'black')
ax1.set_xlabel('$p$')
ax1.set_ylabel('$\Pi(p,L)$')
ax1.legend()

pc=np.average(pcs)
err=np.std(pcs)
print(f'P-c is approximated as {pc:.5} +/- {err:.1}')
plt.show()
