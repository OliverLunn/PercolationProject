import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from main import Percolation2D
import scipy.optimize as opt

def f(x,m,c):
    return m*x+c

percolation2D = Percolation2D()

sizes=[25,50,100,150,200,250,300,400,500,1000]
runs = 5
mass=np.zeros((runs,len(sizes)))
p=0.59274605079210 #p_c
i=0
for size in tqdm(sizes):
    for run in range(runs):
        j=0
        span = False
        while span == False:
            #generate lattice
            lattice = percolation2D.lattice_random(size,0.59)
            labeled_lattice = percolation2D.cluster_search(lattice)
            max_cluster = percolation2D.max_cluster(size,labeled_lattice)
            #test if biggest cluster spans lattice
            perc_x = np.intersect1d(max_cluster[0,:],max_cluster[-1,:])
            perc = perc_x[np.where(perc_x>0)]
            if (len(perc)>0):
                break

        mass[j,i] = np.count_nonzero(max_cluster)
        j+=1
    i+=1
figure = plt.figure()
ax = plt.axes()
mass = np.average(mass,axis=0)
ax.scatter(np.log(sizes),np.log(mass),color='blue',marker='.',label='Data')

ppot,pcov = opt.curve_fit(f,np.log(sizes),np.log(mass))
err = np.sqrt(np.diag(pcov))
print(f'D={ppot[0]:.4} +/- {err[0]:.4}')
ax.plot(np.log(sizes),f(np.log(sizes),*ppot),color='black',label='Fit')

ax.set_xlabel('$log(L)$')
ax.set_ylabel('$log(M(L))$')
plt.legend()
plt.show()

