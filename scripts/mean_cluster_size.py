#import modules
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
from tqdm import tqdm
import scipy.optimize as opt


from main import Percolation2D

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    p1 = 0.59274605079210  #transition prob
    size = 200
    probs = np.arange(0.1,0.9,0.005)
    rep = 3
    avg_size = np.zeros((len(probs), rep))

    for r in tqdm(range(0,rep)):
        i=0
        for p in probs:
            lattice = percolation2d.lattice_random(size, p)
            labeled_lattice = percolation2d.cluster_search(lattice)
            avg_size[i,r] = percolation2d.average_cluster_size(labeled_lattice)

            i += 1
    ydata = np.average(avg_size,axis=1)
    fig,(ax1) = plt.subplots(1,1)

    ax1.plot(probs, ydata, "ko", label="lattice")
    ax1.vlines(p1,0,1, linestyles='--', color='black', label="$p_c$")
    ax1.set_ylabel("Average Cluster Size, $\zeta_p$", fontsize="20")
    ax1.set_xlabel("Probability, $p$", fontsize="20")
    ax1.tick_params(axis="x", labelsize=18)
    ax1.tick_params(axis="y", labelsize=18)
    plt.show()