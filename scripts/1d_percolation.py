import numpy as np
import matplotlib.pyplot as plt

def gamma(probs, L, gamma_array):
    for p in probs:
        gamma = p**L
        gamma_array = np.append(gamma_array, gamma)

    return gamma_array
    
def cluster_size(probs):
    cluster = (p_c - probs)**-1
    return cluster

if __name__ == '__main__':

    gamma_array = []
    probs = np.arange(0.0001,1,0.001)
    p_c = 1

    fig, (ax1,ax2) = plt.subplots(1,2)

    ax1.plot(probs, gamma(probs, 1, gamma_array), label="L=1")
    ax1.plot(probs, gamma(probs, 2, gamma_array), label="L=2")
    ax1.plot(probs, gamma(probs, 3, gamma_array), label="L=3")
    ax1.plot(probs, gamma(probs, 5, gamma_array), label="L=5")
    ax1.plot(probs, gamma(probs, 25, gamma_array), label="L=20")
    ax1.plot(probs, gamma(probs, 100, gamma_array), label="L=100")

    ax2.plot(probs, cluster_size(probs), label="$(p_c-p)^-1$")
    ax2.plot(probs, -1/np.log(probs),"b", label="$-1/ln(p)$")
    ax2.vlines(p_c, -0.5, np.max(cluster_size(probs)), linestyles ="dotted", colors ="k")
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel("$\Pi(p,L)$")
    ax1.set_xlabel("Occupation probabilty, p")
    ax2.set_ylabel("$s_\zeta$")
    ax2.set_xlabel("Occupation probabilty, p")

    plt.show()