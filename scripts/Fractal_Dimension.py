import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from main import Percolation2D
import scipy.optimize as opt

def f(x,m,c):
    return m*x+c

percolation2D = Percolation2D()

sizes=[216,432,648,864,1080,1296]
runs = 10
mass = np.zeros((runs,len(sizes)))
mass_cg = np.zeros((runs,len(sizes)))
mass_bs = np.zeros((runs,len(sizes)))
b = 2
p=0.59274605079210 #p_c
i=0
for size in tqdm(sizes):
    j=0
    for run in range(runs):
        span = False
        while span == False:
            #generate lattice 
            lattice = percolation2D.lattice_random(size,0.59274605079210)
            labeled_lattice = percolation2D.cluster_search(lattice)
            max_cluster = percolation2D.max_cluster(size, labeled_lattice)

            coarse_grained = percolation2D.coarse_graining(3, size, lattice)
            labeled_lattice_cg = percolation2D.cluster_search(coarse_grained)
            max_cluster_cg = percolation2D.max_cluster(size, labeled_lattice_cg)

            block_spin = percolation2D.renorm_group(b, size, lattice)
            labeled_lattice_bs = percolation2D.cluster_search(block_spin)
            max_cluster_bs = percolation2D.max_cluster(int(size/b), labeled_lattice_bs)


            #test if biggest cluster spans lattice
            perc_x = np.intersect1d(max_cluster[0,:],max_cluster[-1,:])
            perc = perc_x[np.where(perc_x>0)]

            perc_x_cg = np.intersect1d(max_cluster_cg[0,:],max_cluster_cg[-1,:])
            perc_cg = perc_x_cg[np.where(perc_x_cg>0)]

            perc_x_bs = np.intersect1d(max_cluster_bs[0,:],max_cluster_bs[-1,:])
            perc_bs = perc_x_bs[np.where(perc_x_bs>0)]
            
            if (len(perc)>0):
                break

        mass[j,i] = np.count_nonzero(max_cluster)
        mass_cg[j,i] = np.count_nonzero(max_cluster_cg)
        mass_bs[j,i] = np.count_nonzero(max_cluster_bs)
        j+=1
    i+=1

fig = plt.figure()
ax1 = plt.axes()

mass = np.average(mass,axis=0)
mass_cg = np.average(mass_cg,axis=0)
mass_bs = np.average(mass_bs,axis=0)


ax1.scatter(np.log(sizes), np.log(mass), color='k', marker='o', label='Lattice')
ax1.scatter(np.log(sizes), np.log(mass_cg), color='k', marker='>', label='Coarse Grain')
ax1.scatter(np.log(sizes), np.log(mass_bs), color='k', marker='D', label='Block Spin')

ppot,pcov = opt.curve_fit(f,np.log(sizes),np.log(mass))
err = np.sqrt(np.diag(pcov))
print(f'D={ppot[0]:.4} +/- {err[0]:.4}')
ax1.plot(np.log(sizes),f(np.log(sizes),*ppot),color='black',label='Fit')

ppot_cg,pcov_cg = opt.curve_fit(f,np.log(sizes),np.log(mass_cg))
err_cg = np.sqrt(np.diag(pcov_cg))
print(f'D_cg={ppot_cg[0]:.4} +/- {err_cg[0]:.4}')
ax1.plot(np.log(sizes),f(np.log(sizes),*ppot_cg),color='black')


ppot_bs,pcov_bs = opt.curve_fit(f,np.log(sizes),np.log(mass_bs))
err_bs = np.sqrt(np.diag(pcov_bs))
print(f'D_cg={ppot_bs[0]:.4} +/- {err_bs[0]:.4}')
ax1.plot(np.log(sizes),f(np.log(sizes),*ppot_bs),color='black')

ax1.set_xlabel('log(L)', fontsize="22")
ax1.set_ylabel('log(M(L))', fontsize="22")
plt.legend()
plt.show()

