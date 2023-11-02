import matplotlib.pyplot as plt
import numpy as np
from main import Percolation2D

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    fig, (ax1,ax2) = plt.subplots(1,2)
    p=0.59274605079210
    interval=0.00025
    probs = np.arange(0.0,1-interval, interval)
    zeta_array = np.array([])
    renorm_array = np.array([])
    ax1.plot(probs, percolation2d.diverge(probs, p, zeta_array))
    ax1.vlines(p,0,1000, linestyles='--', color='black', label="$p_c$")
    ax1.set_ylim(0,200)
    ax1.set_ylabel("Mean cluster size, $\zeta_p$")
    ax1.set_xlabel("Probabilty,p")
    ax2.plot(probs, percolation2d.renorm_group_theory(probs, renorm_array), label="R(p)")
    ax2.plot(probs, probs,"k--", label="p")
    ax2.set_ylabel("R(p)")
    ax2.set_xlabel("Probabilty,p")


    plt.legend()
    plt.show()