import numpy as np
import matplotlib.pyplot as plt
from main import Percolation2D

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    gamma_array = []
    probs = np.arange(0.0001,1,0.001)
    p_c = 1

    fig, (ax1,ax2) = plt.subplots(1,2)

    ax1.plot(probs, percolation2d.gamma(probs, 1, gamma_array), label="L=1")
    ax1.plot(probs, percolation2d.gamma(probs, 2, gamma_array), label="L=2")
    ax1.plot(probs, percolation2d.gamma(probs, 3, gamma_array), label="L=3")
    ax1.plot(probs, percolation2d.gamma(probs, 5, gamma_array), label="L=5")
    ax1.plot(probs, percolation2d.gamma(probs, 25, gamma_array), label="L=20")
    ax1.plot(probs, percolation2d.gamma(probs, 100, gamma_array), label="L=100")

    ax2.plot(probs, percolation2d.cluster_size(probs, p_c), label="$(p_c-p)^-1$")
    ax2.plot(probs, -1/np.log(probs),"b", label="$-1/ln(p)$")
    ax2.vlines(p_c, -0.5, np.max(percolation2d.cluster_size(probs, p_c)), linestyles ="dotted", colors ="k")
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel("$\Pi(p,L)$")
    ax1.set_xlabel("Occupation probabilty, p")
    ax2.set_ylabel("$s_\zeta$")
    ax2.set_xlabel("Occupation probabilty, p")

    plt.show()