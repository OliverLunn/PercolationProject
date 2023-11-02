import matplotlib.pyplot as plt
import numpy as np
from main import Percolation2D

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    fig, (ax1,ax2) = plt.subplots(1,2)

    p=0.59274605079210
    interval=0.00005
    probs = np.arange(0.0,1-interval, interval)
    zeta_array = np.array([])
    renorm_array = np.array([])
    
    ax1.plot(probs, percolation2d.diverge(probs, p, zeta_array))
    ax1.vlines(p,0,10000, linestyles='--', color='black', label="$p_c$")
    ax1.set_ylabel("Mean cluster size, $\zeta_p$", fontsize="22")
    ax1.set_xlabel("Probabilty, $p$", fontsize="22")
    
    ax2.plot(probs, percolation2d.renorm_group_theory(probs, renorm_array), label="R(p) theory")
    ax2.plot(probs, probs,"k--", label="$p=p*$")
    ax2.set_ylabel("$R(p)$", fontsize="22")
    ax2.set_xlabel("Probabilty, $p$", fontsize="22")
    
    ax1.set_yscale("log") 
    ax1.set_ylim(1, 1e3) 
    ax1.tick_params(axis="x", labelsize=18)
    ax2.tick_params(axis="x", labelsize=18)
    ax1.tick_params(axis="y", labelsize=18)
    ax2.tick_params(axis="y", labelsize=18)
    plt.legend(fontsize="18")
    plt.show()