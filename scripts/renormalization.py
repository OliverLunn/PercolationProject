#import modules
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
from tqdm import tqdm
import scipy.optimize as opt


from main import Percolation2D


def sigmoid(x, x0, a, k1, k2):
    y_fit = 1 / (1 + a*np.exp(-k1*(x-x0)) + b*np.exp(-k2*(x-x0)))
    return y_fit

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    p1 = 0.59274605079210  #transition prob
    size, size1 = 80, 40
    b = 2 #renormalization scaling value
    rep = 100
    probs = np.arange(0.1,0.9,0.005)

    avg_size = np.zeros((len(probs), rep))
    avg_size1 = np.zeros((len(probs), rep))
    renorm_array = []

    for r in tqdm(range(0,rep)):
        i=0
        for p in probs:
            lattice = percolation2d.lattice_random(size, p)
            labeled_lattice = percolation2d.cluster_search(lattice)
            avg_size[i,r] = percolation2d.average_cluster_size(lattice, size)
            #lattice_renorm = percolation2d.renorm_group(b, size, lattice)
            #avg_size1[i,r] = percolation2d.average_cluster_size(lattice_renorm)

            i += 1
    ydata = np.average(avg_size,axis=1)
    #ydata1 is renormalized data 
    #ydata1 = np.average(avg_size1,axis=1) / np.max(avg_size1)

    #popt, pcov = opt.curve_fit(sigmoid, probs, ydata, method='lm')
    #popt1, pcov1 = opt.curve_fit(sigmoid, probs, ydata1, method='lm')
    fig, (ax1) = plt.subplots(1,1)

    ax1.plot(probs, ydata, "ko", label="lattice")
    #ax1.plot(probs, ydata1, "k<", label="Renormalised lattice")
    #ax1.plot(probs, probs, "b--", label="$p=p'$")
    #plt.plot(probs, percolation2d.renorm_group_theory(probs, renorm_array), label="R(p)")
    #ax1.plot(probs, sigmoid(probs, *popt1), "g", label="lattice fit")
    #ax1.plot(probs, sigmoid(probs, *popt), "k", label="lattice fit")
    ax1.vlines(p1,0,1, linestyles='--', color='black', label="$p_c$")
    ax1.set_ylabel("Average Cluster Size, $\zeta_p$", fontsize="20")
    ax1.set_xlabel("Probability, $p$", fontsize="20")
    ax1.tick_params(axis="x", labelsize=18)
    ax1.tick_params(axis="y", labelsize=18)

    plt.legend()
    plt.show()