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
        
    return np.average(array, axis=1)
if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    p1 = 0.59274605079210  #transition prob
    size = 200
    probs = np.arange(0.1,0.99,0.01)
    mod_p = np.abs(probs-p1)
    reps = 1

    avg_size_50 = np.zeros((len(probs), reps))
    fig,(ax1,ax2) = plt.subplots(1,2)

    lattice_150 = avg_size_plot(avg_size_50, 150, probs, reps)
    lattice_175 = avg_size_plot(avg_size_50, 175, probs, reps)
    lattice_200 = avg_size_plot(avg_size_50, 200, probs, reps)

    ax1.plot(probs, lattice_150 , "ro",label="L:150")
    ax1.plot(probs, lattice_175, "b<",label="L:175")
    ax1.plot(probs, lattice_200, "kD",label="L:200")

    ax2.plot(np.log(abs(probs-p1))[2*len(probs)//3:], np.log(lattice_150)[2*len(probs)//3:], "ro", label="L:150")
    ax2.plot(np.log(abs(probs-p1))[2*len(probs)//3:], np.log(lattice_175)[2*len(probs)//3:], "b<", label="L:175")
    ax2.plot(np.log(abs(probs-p1))[2*len(probs)//3:], np.log(lattice_200)[2*len(probs)//3:], "kD", label="L:200")

    ax1.vlines(p1,0,size**2, linestyles='--', color='black', label="$p_c$")
    
    ax1.set_ylabel("$\\frac{M_2(p,L)}{M_1(p,L)}$", fontsize="20")
    ax1.set_xlabel("Probability, $p$", fontsize="20")
    
    ax2.set_xlabel("p-pc", fontsize="20")
    ax1.tick_params(axis="x", labelsize=18)
    ax1.tick_params(axis="y", labelsize=18)
    plt.tight_layout()
    plt.legend()
    plt.show()
