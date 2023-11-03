#import modules
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
from tqdm import tqdm
import scipy.optimize as opt


from main import Percolation2D

def avg_size_plot(array, size, probs, reps):

    for r in tqdm(range(0,reps)):
        i=0 
        for p in probs:
            lattice = percolation2d.lattice_random(size, p)
            labeled_lattice = percolation2d.cluster_search(lattice)
            array[i,r] = percolation2d.average_cluster_size(labeled_lattice)
            i += 1
        
    return np.average(array,axis=1)

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    p1 = 0.59274605079210  #transition prob
    size = 200
    probs = np.arange(0.1,0.99,0.01)
    reps = 5

    avg_size_50 = np.zeros((len(probs), reps))
    avg_size_100 = np.zeros((len(probs), reps))
    avg_size_150 = np.zeros((len(probs), reps))
    avg_size_200 = np.zeros((len(probs), reps))

    fig,(ax1) = plt.subplots(1,1)

    ax1.plot(probs, avg_size_plot(avg_size_50, 150, probs, reps), "ro",label="L:150")
    ax1.plot(probs, avg_size_plot(avg_size_50, 175, probs, reps), "b<",label="L:175")
    ax1.plot(probs, avg_size_plot(avg_size_50, 200, probs, reps), "kD",label="L:200")
    
    ax1.vlines(p1,0,size**2, linestyles='--', color='black', label="$p_c$")
    ax1.set_ylabel("$\\frac{M_2(p,L)}{M_1(p,L)}$", fontsize="20")
    ax1.set_xlabel("Probability, $p$", fontsize="20")
    ax1.tick_params(axis="x", labelsize=18)
    ax1.tick_params(axis="y", labelsize=18)

    plt.legend()
    plt.show()
