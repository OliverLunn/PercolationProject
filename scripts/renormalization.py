import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
from tqdm import tqdm
import scipy.optimize as opt
from main import Percolation2D


if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    p = 0.59274605079210  #transition prob
    size, size1 = 50, 25
    b = 2 #renormalization scaling value
    rep = 5
    probs = np.arange(0.05,0.995,0.01)

    avg_sizes = np.zeros((len(probs), rep))
    avg_size1 = np.zeros((len(probs), rep))
    renorm_array = []

    for r in tqdm(range(0,rep)):
        i=0
        for p in probs:
            lattice = percolation2d.lattice_random(size, p)
            labeled_lattice = percolation2d.cluster_search(lattice)
            avg_sizes[i,r] = percolation2d.average_cluster_size(labeled_lattice)
            lattice_renorm = percolation2d.renorm_group(b, size, lattice)
            lattice_renorm_lab = percolation2d.cluster_search(lattice_renorm)
            avg_size1[i,r] = percolation2d.average_cluster_size(lattice_renorm_lab)

            i += 1
    
    ydata = np.average(avg_size1,axis=1) / np.max(avg_size1)
    plt.plot(probs, ydata, "k.", label="simulation results")
    plt.plot(probs, probs, "b--", label="$p=p'$")
    plt.plot(probs, percolation2d.renorm_group_prediction(probs, renorm_array), label="R(p)")
    plt.ylabel("Average Cluster Size, $\zeta_p$")
    plt.xlabel("Probability, $p$")
    plt.legend()
    plt.show()